__author__ = "Ece Calikus"
__email__ = "ece.calikus@hh.se"

import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from scipy.special import erf
from sklearn.ensemble import IsolationForest


class BaseDetector(object):

    def __init__(self, X_train, X_test, y, contextual_attr, behavioral_attr, params):
        self.X_train = X_train
        self.X_test = X_test
        self.y = y
        self.ref_groups = None
        self.contextual_attr = contextual_attr
        self.behavioral_attr = behavioral_attr
        self.params = params
        self.ref_groups_train = None
        self.ref_groups_test = None

    def _find_reference_groups(self):
        X = np.concatenate((self.X_train, self.X_test), axis=0)
        context_vals = X[:, self.contextual_attr]
        initial_centers = kmeans_plusplus_initializer(context_vals, int(
            self.params['Base_Detector']['n_initial_centers'])).initialize()
        xmeans_instance = xmeans(context_vals, initial_centers, int(
            self.params['Base_Detector']['n_initial_centers']), tolerance=0.002)
        xmeans_instance.process()

        self.ref_groups_train = xmeans_instance.predict(context_vals)
        self.ref_groups_test = self.ref_groups_train[(len(self.ref_groups_train) - len(self.X_test)):]

    def compute_anomaly_scores(self):
        self._find_reference_groups()
        X = np.concatenate((self.X_train, self.X_test), axis=0)
        beh_vals_train = X[:, self.behavioral_attr]
        beh_vals_test = self.X_test[:, self.behavioral_attr]
        n_groups = len(set(self.ref_groups_train))
        scores_all = np.empty([len(beh_vals_test)])
        for i in range(n_groups):
            indices_train = np.where(self.ref_groups_train == i)[0]
            indices_test = np.where(self.ref_groups_test == i)[0]
            training_vals = beh_vals_train[indices_train, :]
            test_vals = beh_vals_test[indices_test, :]
            if len(training_vals) == 0:
                print('Empty training cluster')
                continue
            if len(test_vals) == 0:
                print('Empty test cluster')
                continue

            clf = IsolationForest(random_state=47).fit(training_vals)
            training_scores = clf.score_samples(training_vals)
            training_scores = 0 - training_scores
            test_scores = clf.score_samples(test_vals)
            test_scores = 0 - test_scores

            probs = self.unify_scores(test_scores, training_scores)

            for j in range(len(indices_test)):
                scores_all[indices_test[j]] = probs[j]
        return scores_all

    def unify_scores(self, test_scores, training_scores):
        probs = np.zeros([len(test_scores), 2])
        pre_erf_score = (test_scores - np.mean(training_scores)) / (
                np.std(training_scores) * np.sqrt(2))
        erf_score = erf(pre_erf_score)
        probs[:, 1] = erf_score.clip(0, 1).ravel()
        probs[:, 0] = 1 - probs[:, 1]
        return probs[:, 1]
