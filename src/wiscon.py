__author__ = "Ece Calikus"
__email__ = "ece.calikus@hh.se"

from base_detector import BaseDetector
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
import warnings
from pyod.utils.example import visualize
from sklearn.utils import class_weight
from numpy import genfromtxt
import math
import os.path
import configparser


class WisCon(object):
    def __init__(self, params, X, y, features):
        self.params = configparser.ConfigParser()
        self.params.read(params)
        self.features = features
        self.X = X
        self.y = y
        self.y_test = None
        self.y_pool = None
        self.importance_scores = None
        self.anomaly_scores_all = None
        self.final_scores = None

    def _create_contexts(self):
        import itertools as iter
        import random
        n_features = len(self.features)
        combination_list = [i for i in range(n_features)]
        combinations = [iter.combinations(combination_list, n) for n in range(1, len(combination_list))]
        flat_combinations = iter.chain.from_iterable(combinations)
        context_list = list(map(lambda x: [list(x), list(set(combination_list) - set(x))], flat_combinations))
        return context_list

    def _scores_to_predictions(self):
        th = float(self.params['Wiscon']['anomaly_threshold'])
        scores_all = self.anomaly_scores_all.T
        labels_all = []
        for scores in scores_all:
            labels = []
            for score in scores:
                if score >= th:
                    labels.append(1)
                else:
                    labels.append(0)
            labels_all.append(labels)
        labels_all = np.asarray(labels_all)

        return labels_all.T

    def _low_confidence_anomaly_sampling(self, predictions):
        from random import uniform
        from math import exp
        bias_factor = float(self.params['Wiscon']['bias_factor'])
        priority_list = []
        if (self.importance_scores is not None) and all(j < 0 for j in self.importance_scores) is not True:
            weights = self.importance_scores
            for i, w in enumerate(weights):
                if w < 0:
                    weights[i] = 0
            max_scores = np.average(predictions, axis=1, weights=weights)
        else:
            max_scores = np.average(predictions, axis=1)

        max_scores = (1 - abs((2 * max_scores) - 1))
        max_scores = np.clip(max_scores, 0, 1)
        for x in max_scores:
            ui = uniform(0.0, 1.0)
            weight = exp(bias_factor * float(x) * 10 ** 2)
            ki = weight * ui
            priority_list.append(ki)

        sample_index = np.argmax(priority_list)
        if self.y_pool[sample_index] == 0:
            return sample_index, 0

        return sample_index, max_scores[sample_index]

    def _update_importance_scores(self, y, y_predict, y_scores, s_weights=None):
        from sklearn.metrics import recall_score
        importance_scores = []
        y_true = np.asarray(y)
        y_predict = np.asarray(y_predict)
        y_scores = np.asarray(y_scores)
        if sum(s_weights) == 0:
            s_weights = (np.ones(len(y_true)) / len(y_true))
        for i, y_score in enumerate(y_scores.T):
            y_pred = y_predict.T[i]
            incorrect = y_pred != y_true
            estimator_error = np.average(incorrect, weights=s_weights, axis=0)

            if estimator_error == 1:
                importance_score = 0.5 * math.log(1e-10 / estimator_error)
            elif estimator_error == 0:
                importance_score = 0.5 * math.log(1.0 / 1e-10)
            else:
                importance_score = 0.5 * math.log((1.0 - estimator_error) / estimator_error)

            importance_scores.append(importance_score)

        return importance_scores

    def _remove_invalid_contexts(self):
        scores = self.anomaly_scores_all.T
        for i in range(len(scores)):
            result = scores[i]
            if np.isnan(result).any():
                np.delete(self.anomaly_scores_all, i, 1)

    def initialize_ensemble(self):

        if self.params['Wiscon']['precomputed'] == 'True':
            self.anomaly_scores_all = genfromtxt('scores.csv', delimiter=',')
            self.y_test = genfromtxt('labels.txt')
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=float(
                self.params['Wiscon']['train_test_ratio']))
            self.y_test = y_test
            context_list = self._create_contexts()
            self.anomaly_scores_all = np.zeros((len(X_test), len(context_list)), float)
            for i in range(len(context_list)):
                base_detector = BaseDetector(X_train=X_train, X_test=X_test,
                                             contextual_attr=context_list[i][0], behavioral_attr=context_list[i][1],
                                             params=self.params)
                self.anomaly_scores_all[:, i] = base_detector.compute_anomaly_scores()

            np.savetxt("scores.csv", self.anomaly_scores_all, delimiter=",")
            np.savetxt("labels.txt", self.y_test, delimiter=",")
        self._remove_invalid_contexts()
        self.y_pool = np.copy(self.y_test)

    def run_active_learning_schema(self):
        budget = int(self.params['Wiscon']['budget'])
        anomaly_scores_pool = np.copy(self.anomaly_scores_all)
        n_context = len(self.anomaly_scores_all[0])
        sample_weights = []
        queried_labels = []
        queried_scores = []
        queried_predictions = []
        predictions_all = self._scores_to_predictions()
        for i in range(budget):
            index, sample_weight = self._low_confidence_anomaly_sampling(predictions_all)

            sample_weights.append(sample_weight)
            sample = anomaly_scores_pool[index]

            queried_labels.append(self.y_pool[index])
            queried_predictions.append(predictions_all[index])
            queried_scores.append(sample)
            anomaly_scores_pool = np.delete(anomaly_scores_pool, index, axis=0)
            predictions_all = np.delete(predictions_all, index, axis=0)
            self.y_pool = np.delete(self.y_pool, index, axis=0)
            self.importance_scores = self._update_importance_scores(queried_labels, queried_predictions, queried_scores,
                                                                    s_weights=sample_weights)[:]

        return np.asarray(self.importance_scores).reshape(1, n_context)

    def anomaly_score_aggregation(self):
        from sklearn import preprocessing
        importance_weighted_scores = []

        if len(set(self.importance_scores)) <= 1:
            for a in range(len(self.importance_scores)):
                self.importance_scores[a] = 1

        elif all(x < 0 for x in self.importance_scores) is True:
            mn, mx = np.nanmin(self.importance_scores), np.nanmax(self.importance_scores)
            self.importance_scores = (self.importance_scores - mn) / (mx - mn)

        for i, importance_score in enumerate(self.importance_scores):
            if importance_score < 0:
                self.importance_scores[i] = 0
        for i in range(len(self.anomaly_scores_all)):
            nan_vals = np.argwhere(np.isnan(self.anomaly_scores_all[i]))
            if len(nan_vals) > 0:

                new_scores = np.delete(self.anomaly_scores_all[i], nan_vals, axis=0)
                new_weights = np.delete(self.importance_scores, nan_vals, axis=0)
                importance_weighted_score = np.average(new_scores, weights=new_weights)

            else:
                importance_weighted_score = np.average(self.anomaly_scores_all[i], weights=self.importance_scores)
            importance_weighted_scores.append(importance_weighted_score)

        self.final_scores = importance_weighted_scores

    def calculate_performance(self):
        avg_precision = average_precision_score(self.y_test, self.final_scores)
        print('AUC-PR:', avg_precision)
        return avg_precision



