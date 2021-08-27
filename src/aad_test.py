import logging

import numpy as np
import pandas as pd
# from ad_examples.common.utils import configure_logger
from ad_examples.aad.aad_globals import (
    AAD_IFOREST, IFOR_SCORE_TYPE_NEG_PATH_LEN, HST_LOG_SCORE_TYPE, AAD_HSTREES, RSF_SCORE_TYPE,
    AAD_RSFOREST, INIT_UNIF, AAD_CONSTRAINT_TAU_INSTANCE, QUERY_DETERMINISIC, ENSEMBLE_SCORE_LINEAR,
    get_aad_command_args, AadOpts
)
from ad_examples.aad.aad_support import get_aad_model
from ad_examples.aad.query_model import Query
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

"""
A simple no-frills demo of how to use AAD in an interactive loop.
To execute:
pythonw -m ad_examples.aad.demo_aad
"""

logger = logging.getLogger(__name__)


def get_debug_args(budget=100, detector_type=AAD_IFOREST):
    # return the AAD parameters what will be parsed later
    return ["--resultsdir=./temp", "--randseed=42",
            "--reruns=1",
            "--detector_type=%d" % detector_type,
            "--forest_score_type=%d" %
            (IFOR_SCORE_TYPE_NEG_PATH_LEN if detector_type == AAD_IFOREST
             else HST_LOG_SCORE_TYPE if detector_type == AAD_HSTREES
            else RSF_SCORE_TYPE if detector_type == AAD_RSFOREST else 0),
            "--init=%d" % INIT_UNIF,  # initial weights
            "--withprior", "--unifprior",  # use an (adaptive) uniform prior
            # ensure that scores of labeled anomalies are higher than tau-ranked instance,
            # while scores of nominals are lower
            "--constrainttype=%d" % AAD_CONSTRAINT_TAU_INSTANCE,
            "--querytype=%d" % QUERY_DETERMINISIC,  # query strategy
            "--num_query_batch=1",  # number of queries per iteration
            "--budget=%d" % budget,  # total number of queries
            "--tau=0.03",
            # normalize is NOT required in general.
            # Especially, NEVER normalize if detector_type is anything other than AAD_IFOREST
            # "--norm_unit",
            "--forest_n_trees=100", "--forest_n_samples=256",
            "--forest_max_depth=%d" % (100 if detector_type == AAD_IFOREST else 7),
            # leaf-only is preferable, else computationally and memory expensive
            "--forest_add_leaf_nodes_only",
            "--ensemble_score=%d" % ENSEMBLE_SCORE_LINEAR,
            # "--bayesian_rules",
            "--resultsdir=./temp",
            "--log_file=./temp/demo_aad.log",
            "--debug"]


def detect_anomalies_and_describe(x_train, x_test, y_train, y_test, opts):
    rng = np.random.RandomState(opts.randseed)

    # prepare the AAD model
    model = get_aad_model(x_train, opts)
    model.fit(x_train)
    # model.clf.predict()
    model.init_weights(init_type=opts.init)

    # get the transformed data which will be used for actual score computations
    x_transformed_train = model.transform_to_ensemble_features(x_train, dense=False, norm_unit=opts.norm_unit)

    x_transformed_test = model.transform_to_ensemble_features(x_test, dense=False, norm_unit=opts.norm_unit)

    # populate labels as some dummy value (-1) initially
    y_labeled = np.ones(x.shape[0], dtype=int) * -1

    # at this point, w is uniform weight. Compute the number of anomalies
    # discovered within the budget without incorporating any feedback
    baseline_scores = model.get_score(x_transformed_train, model.w)
    baseline_queried = np.argsort(-baseline_scores)
    baseline_found = np.cumsum(y[baseline_queried[np.arange(opts.budget)]])
    print("baseline found:\n%s" % (str(list(baseline_found))))
    print(average_precision_score(y_train, baseline_scores))
    qstate = Query.get_initial_query_state(opts.qtype, opts=opts, budget=opts.budget)
    queried = []  # labeled instances
    ha = []  # labeled anomaly instances
    hn = []  # labeled nominal instances
    while len(queried) < opts.budget:
        ordered_idxs, anom_score = model.order_by_score(x_transformed_train)
        qx = qstate.get_next_query(ordered_indexes=ordered_idxs,
                                   queried_items=queried)
        queried.extend(qx)
        for xi in qx:
            y_labeled[xi] = y_train[xi]  # populate the known labels
            if y_train[xi] == 1:
                ha.append(xi)
            else:
                hn.append(xi)

        # incorporate feedback and adjust ensemble weights
        model.update_weights(x_transformed_train, y_labeled, ha=ha, hn=hn, opts=opts, tau_score=opts.tau)

        # most query strategies (including QUERY_DETERMINISIC) do not have anything
        # in update_query_state(), but it might be good to call this just in case...
        qstate.update_query_state()

    # the number of anomalies discovered within the budget while incorporating feedback
    found = np.cumsum(y_train[queried])
    print("AAD found:\n%s" % (str(list(found))))

    # generate compact descriptions for the detected anomalies

    final_scores = model.get_score(x_transformed_test, model.w)
    # print(average_precision_score(y_test, final_scores))
    # return average_precision_score(y_test, final_scores)
    print(roc_auc_score(y_test, final_scores))
    return roc_auc_score(y_test, final_scores)

    # then create the actual AadOpts from the args


args = get_aad_command_args(debug=True, debug_args=get_debug_args(budget=60))

opts = AadOpts(args)
print('arr 60')
df_feat = pd.read_csv(".../WISCON/datasets/abalone.csv", index_col=None)
y = df_feat.iloc[:, -1].values
x = df_feat.iloc[:, :-1].values

# X_train,X_test,y_train, y_test= generate_data_clusters(n_train=3570, n_test=1530, n_clusters=5, density="different", size="different",n_features=10, contamination=0.02, random_state=5)
# x=np.append(X_train, X_test, axis=0)
# y=np.append(y_train, y_test, axis=0)
scores = []

for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
    # run interactive anomaly detection loop
    scores.append(detect_anomalies_and_describe(x_train, x_test, y_train, y_test, opts))

print(np.mean(scores))
print(np.std(scores))
