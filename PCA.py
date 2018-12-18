import numpy as np
import os
import pandas


# read dataset
# pwd = os.getcwd()
# path = os.path.join(pwd, 'Dataset')
# data1_path = os.path.join(path, "X1_t1.csv")
# dataset = pandas.read_csv(data1_path)

# num_example = dataset.shape[0]
# num_feature = dataset.shape[1] - 1

# dataset = dataset.values
# features = dataset[:, : -1]
# targets = dataset[:, -1]

# test mean
# test_array = np.arange(12)
# test_array = test_array.reshape(4, 3)
# # test_array = np.random.random((4, 3))
# features = test_array
# num_example = 4
# num_feature = 3

def PCA(features, k):
    # read feature
    num_example = features.shape[0]
    num_feature = features.shape[1]
    # mean
    mean_features = np.array([np.mean(features[:, index]) for index in range(num_feature)])
    # print(mean_features)
    norm_features = features - mean_features
    # print(norm_features)

    covariance = np.cov(norm_features.T)
    # print(covariance.shape)

    # EVD
    eig_val, eig_vec = np.linalg.eig(covariance)
    # print(eig_val.shape)
    # print(eig_vec.shape)

    eig_pairs = [(np.abs(eig_val[index]), eig_vec[:, index]) for index in range(len(eig_val))]
    eig_pairs = sorted(eig_pairs, key=lambda x:x[0], reverse=True)
    # print(eig_pairs)

    new_features = np.array([element[1] for element in eig_pairs[: k]])
    reduced_data = np.matmul(new_features, norm_features.T).T
    return reduced_data