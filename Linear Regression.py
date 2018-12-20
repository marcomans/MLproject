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
PCA_k = 6

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


def Evaluating(W, x, targets):
    num_example = x.shape[0]
    x_axis = [i + 1 for i in range(num_example)]
    prediction_list = [np.dot(x[index, :], W1) for index in range(num_example)]
    # plt.scatter(x_axis, prediction_list)
    plt.plot(x_axis, prediction_list,'.b')
    plt.plot(x_axis, targets,'.r')
    for i in range(num_example):
        x = [i+1, i+1]
        plt.plot(x, [prediction_list[i], targets[i]], color='k')
    plt.xlabel('Sample')
    plt.ylabel('Output')
    plt.show()



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
reduced_all_features, new_features = PCA_ed(all_features, PCA_k)
# reduced_all_features = all_features
# print(all_features[0,:])
# print(reduced_all_features[0,:])
np.save("linear_vector.npy", new_features)
# bias
bias = np.ones((num_example, 1))
biased_reduced_all_features = np.hstack((bias, reduced_all_features))
# print(biased_reduced_all_features)

# extract all targets
all_targets = dataset[:, -1]
all_targets = all_targets.reshape((num_example, 1))
# print(all_targets.shape)

# initialize weight

read_file = 1

if read_file:
    W = np.load("Weight{}.npy".format(PCA_k))
else:
    W = np.random.randint(-5, 5, size=[PCA_k + 1,1])
    W = W / 10
    np.save("Weight{}.npy".format(PCA_k), W)


# parameters
learning_rate = 0.000078
num_epoch = 100000
# W1 = Training(W, biased_reduced_all_features[ : -50, :], all_targets[ : -50], learning_rate, num_epoch)
# print(np.dot(biased_reduced_all_features[-1, :], W1))
# print(Criterion(W1, biased_reduced_all_features[-50 : , :], all_targets[-50 :]))
# np.save("Weight{}_{}.npy".format(PCA_k, PCA_k), W1)

W1 = np.load("Weight{}_{}.npy".format(PCA_k, PCA_k))
Evaluating(W1, biased_reduced_all_features[-50 : , :], all_targets[-50 :])
# print(np.dot(biased_reduced_all_features[-4, :], W1))
# print(np.dot(biased_reduced_all_features[-5, :], W1))
# print(np.dot(biased_reduced_all_features[-6, :], W1))

# num_epoch = 10000
# start = 0.00006
# end = 0.00008
# pace = (end - start) / 10
# for mul in range(10):
#     alpha = start + mul * pace
#     W1 = Training(W, biased_reduced_all_features, all_targets, alpha, num_epoch)
#     error = Criterion(W1, biased_reduced_all_features, all_targets)
#     print('alpha: ', alpha, error)
#     print(np.dot(biased_reduced_all_features[-1, :], W1))