import pandas
import os
import math
# import matplotlib
# matplotlib.use('GTKAgg')
import time
import matplotlib.pyplot as plt
import numpy as np
from PCA import PCA_ed
import copy
import math


# parameters
PCA_k = 8

def Criterion(W, x, targets):
    num_example = x.shape[0]
    Error = 0    
    for index in range(num_example):
        temp = (targets[index] - np.dot(W.T, x[index, :])) ** 2
        Error += temp
    return math.sqrt(Error / num_example)


def Updating(W, x, targets, alpha):
    num_example = x.shape[0]
    difference = targets - np.matmul(x, W)
    result = W + np.matmul(x.T, difference) * (2 * alpha / num_example)
    return result


def Training(Weight, x, targets, alpha, num_epoch):
    epoch_list = [epoch for epoch in range(num_epoch)]
    error_list = []
    for epoch in range(num_epoch):
        Weight = Updating(Weight, x, targets, alpha)
        # print('w', W)
        # error = Criterion(W, x, targets)
        # print(error)
        # error_list.append(error)
    # np.save("Weight.npy", W)
    # plt.plot(epoch_list, error_list)
    # plt.savefig('test.jpg')
    # plt.show()
    return Weight


# read dataset
pwd = os.getcwd()
path = os.path.join(pwd, 'Dataset')
data1_path = os.path.join(path, "X1_t1.csv")
dataset = pandas.read_csv(data1_path)
# extract data
num_example = dataset.shape[0]
num_feature = dataset.shape[1] - 1
# extract value
dataset = dataset.values
# extract all features
all_features = dataset[:, : -1]

# PCA
# reduced_all_features = PCA_ed(all_features, PCA_k)
reduced_all_features = all_features
# print(all_features[0,:])
# print(reduced_all_features[0,:])

# bias
bias = np.ones((num_example, 1))
biased_reduced_all_features = np.hstack((bias, reduced_all_features))
# print(biased_reduced_all_features)

# extract all targets
all_targets = dataset[:, -1]
all_targets = all_targets.reshape((num_example, 1))
# print(all_targets.shape)

# parameter
# split_parts = 4
# # split features
# length = int(num_example / split_parts)
# feature_parts = [biased_reduced_all_features[part * length : (part + 1) * length, :] for part in range(split_parts)]
# stacked_features = np.vstack((feature_parts[0], feature_parts[1], feature_parts[2], feature_parts[3]))
# # print(stacked_features.shape)

# # split targets
# target_parts = [all_targets[part * length : (part + 1) * length] for part in range(split_parts)]
# stacked_targets = np.vstack((target_parts[0], target_parts[1], target_parts[2], target_parts[3]))
# # print(stacked_targets.shape)

# initialize weight
# W = np.random.random((8 + 1, 1))

# W = np.ones((PCA_k + 1, 1))
# W = np.random.random((PCA_k + 1, 1))
# W = np.random.randint(-5, 5, size=[PCA_k + 1,1])
# W = W / 10
# np.save("Weight8.npy", W)

W = np.load("Weight8.npy")

# parameters
learning_rate = 0.0000004
num_epoch = 1000
W1 = Training(W, biased_reduced_all_features, all_targets, learning_rate, num_epoch)
print(np.dot(biased_reduced_all_features[-4, :], W1))
print(Criterion(W1, biased_reduced_all_features, all_targets))
np.save("Weight8_8.npy", W1)
# num_epoch = 10000
# start = 0.0000004
# end = 0.0000005
# pace = (end - start) / 10

# for mul in range(10):
#     alpha = start + mul * pace
#     W1 = Training(W, biased_reduced_all_features, all_targets, alpha, num_epoch)

#     error = Criterion(W1, biased_reduced_all_features, all_targets)
#     print('alpha: ', alpha, error)
#     print(np.dot(biased_reduced_all_features[-1, :], W1))