import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.sod import SOD
from pyod.models.rod import ROD
from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.feature_bagging import FeatureBagging
from sklearn import model_selection
from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split
from pyod.utils.data import generate_data, generate_data_clusters

def find_best_params(X_train, y_train, param_grid, clf, metric='AUC-PR'):
    if metric == "AUC-PR":
        scorer = make_scorer(average_precision_score)
    else:
        scorer = make_scorer(roc_auc_score)
    grid_dt_estimator = model_selection.GridSearchCV(clf,
                                                     param_grid,
                                                     scoring=scorer,
                                                     refit=True,
                                                     cv=2, return_train_score=True)
    grid_dt_estimator.fit(X_train, y_train)

    df_res = pd.DataFrame(grid_dt_estimator.cv_results_)
    x = df_res[["mean_test_score", "params"]].sort_values(by=["mean_test_score"], ascending=False).head()
    # print(x.mean_test_score)
    print(x)
    optimal_forest = grid_dt_estimator.best_estimator_
    return optimal_forest.get_params(), optimal_forest


def test_baselines(dataset="annthyroid.csv", baseline="IForest", params=None, metric='AUC-PR'):
    df_feat = pd.read_csv(dataset, index_col=None)
    y_feat = df_feat.iloc[:, -1].values
    X_feat = df_feat.iloc[:, :-1].values
    # X_feat = X_feat.to_numpy()

    # X_train,X_test,y_train, y_test= generate_data_clusters(n_train=3535, n_test=1515, n_clusters=5, density="different", size="different",n_features=10, contamination=0.02, random_state=5)
    # X_feat=np.append(X_train, X_test, axis=0)
    # y_feat=np.append(y_train, y_test, axis=0)

    auc = []

    for i in range(10):
        X_train_original, X_test_original, y_train, y_test = train_test_split(X_feat, y_feat,
                                                                              test_size=0.3, stratify=y_feat)
        contam = sum(y_train) / len(y_train)

        if baseline == "IForest":
            clf = IForest(contamination=contam)
        elif baseline == "LOF":
            clf = LOF(contamination=contam).fit(X_train_original)
        elif baseline == "OCSVM":
            clf = OCSVM(contamination=contam).fit(X_train_original)
        elif baseline == "LODA":
            clf = LODA(contamination=contam).fit(X_train_original)
        elif baseline == "SOD":
            clf = SOD(contamination=contam, n_neighbors=200, ref_set=80, alpha=0.8).fit(X_train_original)
        elif baseline == "FB":
            clf = FeatureBagging(contamination=contam).fit(X_train_original)
        elif baseline == "COPOD":
            clf = COPOD(contamination=contam).fit(X_train_original)
        elif baseline == "ROD":
            clf = ROD(contamination=contam).fit(X_train_original)

        elif baseline == "ABOD":
            clf = ABOD(contamination=contam,n_neighbors=40, method='fast').fit(X_train_original)

        if params is None:
            params, clf = find_best_params(X_train_original, y_train, get_param_grid(baseline), clf, metric='AUC-PR')

        scores = clf.predict_proba(X_test_original, method='unify')[:, 1]

        if metric == "AUC-PR":
            r = average_precision_score(y_test, scores)
            print(baseline, r)
            auc.append(r)
        else:
            r = roc_auc_score(y_test, scores)
            print(baseline, r)
            auc.append(r)

    print('mean performance', np.mean(auc))
    print('std performance', np.std(auc))

x=list(10. ** np.arange(-5, 4))
def get_param_grid(baseline):
    if baseline == "IForest":
        return {'n_estimators': list(range(100, 500, 100)),
                'max_features': list(np.arange(0.1, 1, 0.1))}

    elif baseline == "LOF":
        return {'n_neighbors': list(range(10, 200, 10))}

    elif baseline == "OCSVM":
        return {'kernel': ['poly','rbf','sigmoid'],
                'gamma': list(10. ** np.arange(-3, 3)),
                'nu': [0.01, 0.5, 0.99]}

    elif baseline == "LODA":
        return {'n_bins': list(range(2, 100, 2)),
                'n_random_cuts': list(range(40, 500, 20))}
    elif baseline == "SOD":
        return {'n_neighbors': list(range(10, 200, 20)),
                'ref_set':  list(range(10, 100, 10)),
                'alpha': list(np.arange(0.2, 1, 0.1))}
    elif baseline == "FB":
        return {'n_estimators': list(range(100, 500, 100)),
                'max_features': list(np.arange(0.2, 1, 0.1))}
    elif baseline == "ABOD":
        return {'n_neighbors': list(range(10, 10, 20))}


print("cardio", "OC")
test_baselines("/Users/ececal/PycharmProjects/WISCON/datasets/arrhythmia_pca210.csv", "ROD", params=1,
               metric='AUC-PR')
print("thy", "FB")
