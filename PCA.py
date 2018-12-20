import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# read dataset
pwd = os.getcwd()
path = os.path.join(pwd, 'Dataset')
data1_path = os.path.join(path, "X1_t1.csv")
dataset = pandas.read_csv(data1_path)

num_example = dataset.shape[0]
num_feature = dataset.shape[1] - 1

dataset = dataset.values
features = dataset[:, : -1]
targets = dataset[:, -1]


def PCA_ed(features, k, test=False):
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
    # print(eig_val)
    eig_pairs = [(np.abs(eig_val[index]), eig_vec[:, index]) for index in range(len(eig_val))]
    eig_pairs = sorted(eig_pairs, key=lambda x:x[0], reverse=True)
    # print(eig_pairs)

    new_features = np.array([element[1] for element in eig_pairs[: k]])
    reduced_data = np.matmul(new_features, norm_features.T).T
    if test == False:
        return reduced_data, new_features
    else:
        return eig_pairs

eig_v, plt_2d, plt_1d = 0, 0, 0



if eig_v:
    eig_pairs = PCA_ed(features, 1, test=True)
    for pair in eig_pairs:
        print(pair[0])

if plt_2d:
    reduced, a = PCA_ed(features, 2)
    x = reduced[:, 0]
    y = reduced[:, 1]
    z = targets
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

if plt_1d:
    reduced, a = PCA_ed(features, 1)
    x = reduced
    y = targets
    plt.scatter(x, y)
    plt.show()