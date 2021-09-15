import random

import numpy as np
import pandas as pd
from scipy.spatial import distance

RS = 123


def generate_geometric_distribution(n_gaussian, p):
    from scipy.stats import geom
    geometric_dist = np.zeros((n_gaussian, n_gaussian))
    probs = geom.pmf([i + 1 for i in range(n_gaussian)], p=p)
    for i in range(n_gaussian):
        np.random.shuffle(probs)
        geometric_dist[i] = probs
    return probs


def perturp_one_record(x_val, y_val, perturbed_con, perturbed_beh, k):
    y_val = list(y_val)
    random_sample_index = list(
        np.random.randint(0, len(perturbed_con), k))

    x_values = perturbed_con[random_sample_index]
    y_values = perturbed_beh[random_sample_index]

    distances = []
    for y_value in y_values:
        y_value = list(y_value)
        distances.append(distance.euclidean(y_value, y_val))

    index_to_perturb_with = np.argmax(distances)

    old_value = x_val
    perturbed_value = np.array([[old_value, y_values[index_to_perturb_with]]])
    original_value = np.array([[x_values[index_to_perturb_with], y_values[index_to_perturb_with]]])

    return y_values[index_to_perturb_with], perturbed_value, original_value


def make_dataset_multi_context():
    probs = generate_geometric_distribution(5, 0.6)
    df1= make_dataset_single(500, probs, 6, 4, 5)
    df2 =make_dataset_single(300, probs, 7, 3, 5)
    df3 =make_dataset_single(200,probs, 9,1,5)

    frames = [df1, df2, df3]

    result = pd.concat(frames)
    result.to_csv('synthetic_multi.csv', index=False)


def make_dataset_single(n_samples, prob, num_of_con_features, num_of_beh_features, num_of_gaussians):
    from sklearn.metrics import pairwise_distances
    U = []
    V = []
    means_con = []
    cov_matrixes_con = []
    means_beh = []
    cov_matrixes_beh = []
    for i in range(num_of_gaussians):
        means_con.append(np.random.uniform(0, 30, num_of_con_features))

    distances = pairwise_distances(means_con, Y=None, metric='euclidean')
    distances_avg = np.average(distances, axis=0)
    diag_vals_con = distances_avg

    for i in range(num_of_gaussians):
        cov_matrix = np.eye(num_of_con_features, dtype=int) * diag_vals_con[i]
        cov_matrixes_con.append(cov_matrix)

    for i in range(num_of_gaussians):
        u = np.random.multivariate_normal(means_con[i], cov_matrixes_con[i], n_samples)
        U.append(u)

    for i in range(num_of_gaussians):
        means_beh.append(np.random.uniform(0, 30, num_of_beh_features))

    distances = pairwise_distances(means_beh, Y=None, metric='euclidean')
    distances_avg = np.average(distances, axis=0)
    diag_val_beh = distances_avg

    for i in range(num_of_gaussians):
        cov_matrix = np.eye(num_of_beh_features, dtype=int) * diag_val_beh[i]
        cov_matrixes_beh.append(cov_matrix)

    for i in range(num_of_gaussians):
        v = np.random.multivariate_normal(means_beh[i], cov_matrixes_beh[i], n_samples)
        V.append(v)

    D = []

    map_index = np.argsort(prob)
    con_features2 = []
    beh_features2 = []

    for i in range(num_of_gaussians):
        u = U[i]
        v = V[map_index[i]]
        for x in u:
            rand = random.randint(0, len(v) - 1)
            y = v[rand]
            D.append([x, y])
            con_features2.append(x)
            beh_features2.append(y)
    con_features = np.asarray(con_features2, dtype=np.float32)
    beh_features = np.asarray(beh_features2, dtype=np.float32)

    contextual_anomalies_con, contextual_anomalies_beh = create_contextual_anomalies(con_features, beh_features,
                                                                                     num_of_gaussians)

    con_features_all = np.concatenate((con_features, contextual_anomalies_con), axis=0)

    beh_features_all = np.concatenate((beh_features, contextual_anomalies_beh), axis=0)

    features = np.concatenate((con_features_all, beh_features_all), axis=1)

    ground_truth = [0 for i in range(len(con_features))]
    for a in range(len(contextual_anomalies_con)):
        ground_truth.append(1)

    df = pd.DataFrame(data=features[0:, 0:], index=[i for i in range(features.shape[0])])
    df['ground_truth'] = ground_truth
    print(df.head())

    df.to_csv('synthetic_single.csv', index=False)
    return df


def create_contextual_anomalies(con_features, beh_features, n_gaussian):
    from sklearn.model_selection import train_test_split
    from sklego.mixture import GMMOutlierDetector

    train_data_con, test_data_con, train_data_beh, test_data_beh = train_test_split(con_features, beh_features,
                                                                                    test_size=0.2)

    mod = GMMOutlierDetector(n_components=n_gaussian, threshold=0.8).fit(train_data_con)

    x = mod.predict(test_data_con)

    indices = np.where(x != 1)[0]
    indices2 = np.where(x == 1)[0]

    outliers_con = test_data_con[indices]
    non_outliers_con = test_data_con[indices2]

    outliers_beh = test_data_beh[indices]
    non_outliers_beh = test_data_beh[indices2]

    print(len(non_outliers_con))
    print(len(outliers_con))

    print(len(non_outliers_beh))
    print(len(outliers_beh))

    non_perturbed_con, perturbed_con, non_perturbed_beh, perturbed_beh = train_test_split(non_outliers_con,
                                                                                          non_outliers_beh,
                                                                                          test_size=int(
                                                                                              len(test_data_con) * 0.5))

    non_perturbed_con = np.concatenate((outliers_con, non_perturbed_con), axis=0)
    non_perturbed_beh = np.concatenate((outliers_beh, non_perturbed_beh), axis=0)

    print(len(non_perturbed_con))
    print(len(perturbed_con))

    print(len(non_perturbed_beh))
    print(len(perturbed_beh))

    normal_con, before_perturbed_con, normal_beh, before_perturbed_beh = train_test_split(perturbed_con, perturbed_beh,
                                                                                          test_size=int(
                                                                                              len(con_features) * 0.02))

    contextual_anomalies_con = np.zeros((len(before_perturbed_con), len(con_features[0])), float)

    contextual_anomalies_beh = np.zeros((len(before_perturbed_beh), len(beh_features[0])), float)

    for i in range(len(before_perturbed_beh)):
        k = min(50, int(len(perturbed_con) / 4))
        a, b, c = perturp_one_record(before_perturbed_con[i], before_perturbed_beh[i], perturbed_con, perturbed_beh, k)

        contextual_anomalies_con[i, :] = b[0][0]
        contextual_anomalies_beh[i, :] = b[0][1]

    print(len(contextual_anomalies_con))

    return contextual_anomalies_con, contextual_anomalies_beh
