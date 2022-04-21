from math import log, exp

from scipy.special import erf
# from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from IsolationForestContextual import IsolationForest


# --[Basic Function]---------------------------------------------------------------------
# input decision_values, real_labels{1,-1}, #positive_instances, #negative_instances
# output [A,B] that minimize sigmoid likilihood
# refer to Platt's Probablistic Output for Support Vector Machines
def SigmoidTrain(deci, label, A=None, B=None, prior0=None, prior1=None):
    # Count prior0 and prior1 if needed
    if prior1 == None or prior0 == None:
        prior1, prior0 = 0, 0
        for i in range(len(label)):
            if label[i] > 0:
                prior1 += 1
            else:
                prior0 += 1

    # Parameter Setting
    maxiter = 1000  # Maximum number of iterations
    minstep = 1e-10  # Minimum step taken in line search
    sigma = 1e-12  # For numerically strict PD of Hessian
    eps = 1e-5
    length = len(deci)

    # Construct Target Support
    hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
    # print(hiTarget)
    loTarget = 1 / (prior0 + 2.0)
    length = prior1 + prior0
    t = []

    for i in range(length):
        if label[i] > 0:
            t.append(hiTarget)
        else:
            t.append(loTarget)

    # print(np.mean(t))
    # Initial Point and Initial Fun Value
    A, B = 0.0, log((prior0 + 1.0) / (prior1 + 1.0))
    # print("A,B",A,B)
    fval = 0.0

    for i in range(length):
        fApB = deci[i] * A + B

        if fApB >= 0:  # Positive class hence label will be +1
            fval += t[i] * fApB + log(1 + exp(-fApB))
        else:  # Negative class label will be -1
            fval += (t[i] - 1) * fApB + log(1 + exp(fApB))

    for it in range(maxiter):
        # Update Gradient and Hessian (use H' = H + sigma I)
        h11 = h22 = sigma  # Numerically ensures strict PD
        h21 = g1 = g2 = 0.0
        for i in range(length):
            fApB = deci[i] * A + B
            if (fApB >= 0):
                p = exp(-fApB) / (1.0 + exp(-fApB))
                q = 1.0 / (1.0 + exp(-fApB))
            else:
                p = 1.0 / (1.0 + exp(fApB))
                q = exp(fApB) / (1.0 + exp(fApB))
            d2 = p * q
            h11 += deci[i] * deci[i] * d2
            h22 += d2
            h21 += deci[i] * d2
            d1 = t[i] - p
            g1 += deci[i] * d1
            g2 += d1

        # Stopping Criteria
        if abs(g1) < eps and abs(g2) < eps:
            break

        # Finding Newton direction: -inv(H') * g
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB
        # Line Search
        stepsize = 1
        while stepsize >= minstep:
            newA = A + stepsize * dA
            newB = B + stepsize * dB

            # New function value
            newf = 0.0
            for i in range(length):
                fApB = deci[i] * newA + newB
                if fApB >= 0:
                    newf += t[i] * fApB + log(1 + exp(-fApB))
                else:
                    newf += (t[i] - 1) * fApB + log(1 + exp(fApB))

            # Check sufficient decrease
            if newf < fval + 0.0001 * stepsize * gd:
                A, B, fval = newA, newB, newf
                break
            else:
                stepsize = stepsize / 2.0

        if stepsize < minstep:
            print("line search fails", A, B, g1, g2, dA, dB, gd)

            return [A, B]

    if it >= maxiter - 1:
        print("reaching maximal iterations", g1, g2)
    return (A, B, fval)


def SigmoidPredict(deci, AB):
    A, B = AB
    fApB = deci * A + B
    if (fApB >= 0):
        return exp(-fApB) / (1.0 + exp(-fApB))
    else:
        return 1.0 / (1 + exp(fApB))
    return prob


def Expectation(score, A, B):
    t = []
    for i in range(len(score)):
        if A * score[i] + B >= 0:
            t.append(1)
        else:
            t.append(0)
    p = np.mean(t)
    t[t == 0] = -1
    # print(t)
    return t


def EM(score, A_init, B_init, prior0, prior1, maxit=1000, tol=1e-8):
    # Estimation of parameter(Initial)
    flag = 0
    A_cur = A_init
    B_cur = B_init
    A_new = 0.0
    B_new = 0.0

    # Iterate between expectation and maximization parts

    for i in range(maxit):
        # print(i)
        if (i != 0):
            (A_new, B_new) = SigmoidTrain(score, Expectation(score, A_cur, B_cur), A_cur, B_cur)
            # print(A_new, B_new)
        else:
            t = []
            for i in range(len(score)):
                if A_cur * score[i] + B_cur >= 0:
                    t.append(1)
                else:
                    t.append(0)
            t[t == 0] = -1
            (A_new, B_new) = SigmoidTrain(score, t, A_cur, B_cur, prior0, prior1)
            # print(A_new, B_new)

        # Stop iteration if the difference between the current and new estimates is less than a tolerance level
        if (A_cur - A_new < tol and B_cur - B_new < tol):
            flag = 1
            # break
        # Otherwise continue iteration
        A_cur = A_new
        B_cur = B_new
    if (not flag):
        print("Didn't converge\n")

    return (A_cur, B_cur)


def SigmoidFitting(score, proportion):
    fval = []
    A = []
    B = []
    sorted_score = list(sorted(set(np.round(score, decimals=2))))
    for i in range(int(proportion * len(sorted_score))):
        threshold = sorted_score[i]
        # print(threshold)
        t = [1 if j <= threshold else -1 for j in score]
        (a, b, f) = SigmoidTrain(score, t)
        A.append(a)
        B.append(b)
        fval.append(f)
    return (A, B, fval)


def SigmoidFittingGrid(score, proportion):
    ngrid = score.shape[1]
    fval = []
    A = []
    B = []
    threshold = []
    for param in range(ngrid):
        a, b, f = SigmoidFitting(score[:, param], proportion)
        fval.append(min(f))
        A.append(a[np.argmin(f)])
        B.append(b[np.argmin(f)])
        threshold.append(score[np.argmin(f), param])
    return A, B, fval, threshold


def ContextualForest(contexts, features, features_cat, features_num, gamma_range, ncontexts=None):
    if len(gamma_range) ==0 :
        X_train = features
        X_test = features
        y_test = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=0.3,
                                                            stratify=labels)

    # contexts = shuffle(contexts)
    if (not ncontexts):
        ncontexts = contexts.shape[0]
    context_scores = np.zeros((ncontexts, X_test.shape[0], len(gamma_range)))
    for i in range(ncontexts):
        context = list(contexts.iloc[i,])
        # print(context)
        feature_names = list(features)
        context_features = list()
        behavioral_features = list()

        for feat in feature_names:
            c = feat.split("_")
            if len(set(c).intersection(set(context))) > 0 or feat in context:
                context_features.append(feat)
            else:
                behavioral_features.append(feat)
        context_f = features[context_features]
        behav_f_train = X_train[behavioral_features]
        behav_f_test = X_test[behavioral_features]

        # Finding the categorical and numerical features in context features.
        if (features_cat != None):
            cat_names = list(features_cat)
            cat_context = list(set(context_f).intersection(set(cat_names)))
            context_f_cat = context_f[cat_context]
            num_names = list(features_num)
            num_context = list(set(context_f).intersection(set(num_names)))
            context_f_num = context_f[num_context]
            # Finding the distances of the context space

            cat_context_distance = metrics.pairwise.cosine_similarity(np.array(context_f_cat))
            # Scaling the numerical data using MaxAbsScaler
            context_f_num_scaled = MaxAbsScaler().fit_transform(np.array(context_f_num))
            # Zero mean and unit variance scaling
            # context_f_num_scaled = preprocessing.scale(context_f_num)
            # num_context_distance = metrics.pairwise.euclidean_distances(context_f_num_scaled)

            # print("Cat distance",cat_context_distance)

            for gamma in range(len(gamma_range)):
                print(gamma_range[gamma])
                num_context_distance = metrics.pairwise.rbf_kernel(context_f_num_scaled, gamma=gamma_range[gamma])
                # print("Num distance",num_context_distance)
                context_distance = np.minimum(cat_context_distance, num_context_distance)
                # context_distance = num_context_distance
                rng = np.random.RandomState(42)
                clf = IsolationForest(max_samples=256, random_state=rng, smoothing=True)
                clf.fit(behav_f_train, context_distance)
                context_scores[i, :, gamma] = clf.decision_function(behav_f_test, distance=context_distance)
        else:
            # num_names = list(features_num)
            # num_context = list(set(context_f).intersection(set(num_names)))
            # context_f_num = context_f[num_context]
            # Finding the distances of the context space

            # Scaling the numerical data using MaxAbsScaler
            context_f_num_scaled = MaxAbsScaler().fit_transform(np.array(context_f))
            # Zero mean and unit variance scaling
            # context_f_num_scaled = preprocessing.scale(context_f_num)
            # num_context_distance = metrics.pairwise.euclidean_distances(context_f_num_scaled)

            # print("Cat distance",cat_context_distance)

            for gamma in range(len(gamma_range)):
                # print(gamma_range[gamma])
                num_context_distance = metrics.pairwise.rbf_kernel(context_f_num_scaled, gamma=gamma_range[gamma])
                # print("Num distance",num_context_distance)
                # context_distance = np.minimum(cat_context_distance,num_context_distance)
                context_distance = num_context_distance
                rng = np.random.RandomState(42)
                clf = IsolationForest(max_samples=256, random_state=rng, smoothing=True)
                clf.fit(behav_f_train, context_distance)
                context_scores[i, :, gamma] = clf.decision_function(behav_f_test, distances=context_distance)

    return context_scores, y_test


def aggregate_scores(scores_all):
    fscores = (scores_all - np.amin(scores_all, axis=1, keepdims=True)) / (
                 np.amax(scores_all, axis=1, keepdims=True) - np.amin(scores_all, axis=1,
                                                                     keepdims=True))
    fscores = np.min(fscores, axis=0)
    #fscores = np.median(scores_all, axis = 0)
    #fscores = (fscores - np.amin(fscores, axis = 0))/(np.amax(fscores,axis =0) - np.amin(fscores,axis = 0))
    fscores = np.array(fscores)

    return fscores


def unify_scores(all_scores):
    probs = np.zeros([len(all_scores), len(all_scores[0])])
    for i, scores in enumerate(all_scores):
        scores = -1 * scores
        pre_erf_score = (scores - np.mean(scores)) / (
                np.std(scores) * np.sqrt(2))
        erf_score = erf(pre_erf_score)
        # a=erf_score.clip(0, 1).ravel()
        probs[i, :] = erf_score.clip(0, 1).ravel().tolist()
        # probs[:, 0] = 1 - probs[:, 1]
    return probs


def choose_gamma(gamma_scores_all):
    ABft = SigmoidFittingGrid(gamma_scores_all, proportion=0.1)
    print(ABft)
    A, B, f, t = ABft
    A, B, fmin = A[np.argmin(f)], B[np.argmin(f)], min(f)
    print("A:", A, "B:", B, "f:", f)
    print("Threshold:", t[np.argmin(f)])
    print("Gamma Chosen:", gamma_range[np.argmin(f)])
    fgscores = gamma_scores_all[:, np.argmin(f)]
    return fgscores


# Loading the feature matrix and the context from the UnifiedMeasure.

import pandas as pd
import numpy as np
# pandas2ri.activate()
import pyreadr


from pyod.utils.data import generate_data, generate_data_clusters

# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# readRDS = robjects.r['readRDS']

# Feat matrix 2 will need to be saved as data frame in the R script.
# readRDS = robjects.r['readRDS']
# df_feat = readRDS('ConOut/contexts/features_mamm.RDS')
# df_feat = pandas2ri.ri2py_dataframe(df_feat)
# df_feat = pd.DataFrame(np.transpose(df_feat))
print('syn 3 aucpr')
df_feat = pyreadr.read_r('/Users/ececal/PycharmProjects/WISCON/baselines/ConOut/contexts/features_synthetic32.RDS')
df_feat = df_feat[None]
# Contexts are being stored as data frames in R.-
# df_contexts = readRDS('ConOut/contexts/context_mamm.RDS')
# df_contexts = pandas2ri.ri2py_dataframe(df_contexts)
# df_contexts = pd.DataFrame(df_contexts)

#df_contexts = pyreadr.read_r('/Users/ececal/PycharmProjects/WISCON/baselines/ConOut/contexts/context_syn3.RDS')
#df_contexts = df_contexts[None]
# df_contexts = pd.DataFrame([['V1','V2']], columns={'Var1','Var2'})
# df_contexts = pd.DataFrame([['V1','V2', 'V3']], columns={'Var1','Var2','Var3'})
# df_contexts = pd.DataFrame([['V1','V2', 'V3', 'V4']], columns={'Var1','Var2','Var3','Var4'})
#df_contexts2 = pd.DataFrame([['V1','V2', 'V3', 'V4','V5']], columns={'Var1','Var2','Var3','Var4','Var5'})
# df_contexts = pd.DataFrame([['V1','V2', 'V3', 'V4','V5','V6']], columns={'Var1','Var2','Var3','Var4','Var5','Var6'})
# df_contexts = pd.DataFrame([['V1','V2', 'V3', 'V4','V5','V6','V7']], columns={'Var1','Var2','Var3','Var4','Var5','Var6','Var7'})
# df_contexts = pd.DataFrame([['V1','V2', 'V3', 'V4','V5','V6','V7','V8']], columns={'Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8'})
#df_contexts=df_contexts.append(df_contexts2)
# f_contexts=df_contexts.append(df_contexts3)
# df_contexts = pd.DataFrame([['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
# 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25']],
# columns={'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10',
# 'Var11', 'Var12', 'Var13', 'Var14', 'Var15', 'Var16', 'Var17', 'Var18','Var19', 'Var20',
# 'Var21', 'Var22', 'Var23', 'Var24', 'Var25'})


df_contexts = pd.read_csv("/Users/ececal/PycharmProjects/WISCON/baselines/ConOut/contexts/context_syn3.csv",
index_col=None)
# df_contexts = df_contexts.append({'Var1': 'V1', 'Var2':'V2', 'Var3':'V3','Var4': 'V4', 'Var5':'V5', 'Var6':'V6','Var7':'V7', 'Var8':'V8'}, ignore_index=True)


# Loading typevar
# df_typevar = readRDS('ConOut/contexts/typevar_mamm.RDS')
# df_typevar = pandas2ri.ri2py_dataframe(df_typevar)

# df_typevar = pyreadr.read_r('/Users/ececal/PycharmProjects/WISCON/baselines/ConOut/contexts/typevar_v.RDS')
# df_typevar = df_typevar[None]

# Loading ground truth
# labels = readRDS('ConOut/contexts/labels_mamm.RDS')
# labels = pandas2ri.ri2py(labels)
labels = pyreadr.read_r('/Users/ececal/PycharmProjects/WISCON/baselines/ConOut/contexts/labels_synthetic3.RDS')
labels = labels[None]

# print(df_contexts.head(10))
# Getting dummies for isolation forest input
# test = pd.get_dummies(df_feat)


# categorical = df_typevar[df_typevar['typevar'] == "categorical"].index.tolist()
# other = df_typevar[df_typevar['typevar'] != "categorical"].index.tolist()
# adjusting for indices in py
# categorical = [i-1 for i in categorical]
# other = [i-1 for i in other]

# Handling for no categorical features.
df_feat_other = df_feat
df_feat_all = df_feat
df_feat_cat = None
# The full feature list in built and ready to be passed to iForest.

print(df_feat_all.head(10))
#gamma_range = [0.0001,0.001,0.01,0.1,1,10,100,1000]
gamma_range=[0.0001,0.001]
'''

scores,y_test = ContextualForest(df_contexts, df_feat_all, df_feat_cat, df_feat_other,
                          gamma_range = gamma_range)

fscores= aggregate_scores(scores)
fgscores= choose_gamma(fscores)
gamma_best=0
perf_best=0

for i in range(len(gamma_range)):
    precision, recall, _ = precision_recall_curve(y_test, -1*fscores[:,i])
    aucpr= auc(recall, precision)
    lab = 'Gamma: %f AUC=%.4f' % (gamma_range[i], aucpr)
    print(lab)
    if aucpr>perf_best:
        perf_best=aucpr
        gamma_best=gamma_range[i]

print(gamma_best)
'''

# Now run 10 times with best gamma
perfs = []
perfs_roc = []
# gamma_range=[gamma_best]
for i in range(10):
    scores, y_test = ContextualForest(df_contexts, df_feat_all, df_feat_cat, df_feat_other,
                                      gamma_range=gamma_range)

    gamma_best = 0
    gamma_best_roc = 0
    perf_best = 0
    perf_roc_best = 0
    fscores = aggregate_scores(scores)
    for i in range(len(gamma_range)):
        # agg_scores= aggregate_scores(fscores[:, i])
        # fscores = aggregate_scores(scores[:, :, i])
        precision, recall, _ = precision_recall_curve(y_test, -1 * fscores[:, i])
        auc_pr = auc(recall, precision)
        auc_roc=roc_auc_score(y_test, -1 * fscores[:, i])
        lab = 'Gamma: %f AUC-PR=%.4f' % (gamma_range[i], auc_pr)
        lab2 = 'Gamma: %f AUC-ROC=%.4f' % (gamma_range[i], auc_roc)
        print(lab)
        print(lab2)
        if auc_pr > perf_best:
            perf_best = auc_pr
            gamma_best = gamma_range[i]

        if auc_roc > perf_roc_best:
            perf_roc_best = auc_roc
            gamma_best_roc = gamma_range[i]

    # precision, recall, _ = precision_recall_curve(y_test, -1*fscores[:,0])
    # aucpr = auc(recall, precision)
    perfs.append(perf_best)
    perfs_roc.append(perf_roc_best)
    # print(aucpr)

print(np.mean(perfs))
print(np.std(perfs))
print(np.mean(perfs_roc))
print(np.std(perfs_roc))
