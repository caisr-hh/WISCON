__author__ = "Ece Calikus"
__email__ = "ece.calikus@hh.se"

import numpy as np
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyod.models.iforest import IForest


class BaseDetector(object):

    def __init__(self, X_train, X_test, contextual_attr, behavioral_attr, params):
        self.X_train = X_train
        self.X_test = X_test
        self.contextual_attr = contextual_attr
        self.behavioral_attr = behavioral_attr
        self.params = params
        self.ref_groups_train = None
        self.ref_groups_test = None

    def _find_reference_groups(self):
        context_vals_train = self.X_train[:, self.contextual_attr]
        context_vals_test = self.X_test[:, self.contextual_attr]
        initial_centers = kmeans_plusplus_initializer(context_vals_train, int(
            self.params['Base_Detector']['n_initial_centers'])).initialize()
        xmeans_instance = xmeans(context_vals_train, initial_centers, int(self.params['Base_Detector']['max_clusters']))
        xmeans_instance.process()

        ref_groups_train = xmeans_instance.predict(context_vals_train)
        self.ref_groups_train = ref_groups_train
        ref_groups_test = xmeans_instance.predict(context_vals_test)
        self.ref_groups_test = ref_groups_test

    def compute_anomaly_scores(self):
        self._find_reference_groups()
        beh_vals_train = self.X_train[:, self.behavioral_attr]
        beh_vals_test = self.X_test[:, self.behavioral_attr]
        n_groups = len(set(self.ref_groups_train))
        scores_all = np.empty([len(beh_vals_test)])
        for i in range(n_groups):
            indices_train = np.where(self.ref_groups_train == i)[0]
            indices_test = np.where(self.ref_groups_test == i)[0]
            training_vals = beh_vals_train[indices_train, :]
            test_vals = beh_vals_test[indices_test, :]
            if (len(training_vals) == 0) | (len(test_vals) == 0):
                print('Empty cluster')
                continue
            clf = IForest(random_state=47, contamination=0.01, behaviour='new').fit(training_vals)
            scores_unified = clf.predict_proba(test_vals, method='unify')[:, 1]
            for j in range(len(indices_test)):
                scores_all[indices_test[j]] = scores_unified[j]
        return scores_all
