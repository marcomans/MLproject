import numpy as np
import pandas
import os
import heapq
from PCA import PCA_ed
import matplotlib.pyplot as plt
import math


# parameter
PCA_k = 6


def Prediction(node_list, targets):
    summation = 0
    for index in range(len(node_list)):
        summation += targets[node_list[index]]
    return summation / len(node_list)


def KNN(aim, features, targets, k):
    distance_list = []
    node_list = []
    # get all distrances
    for index in range(features.shape[0]):
        compare = features[index, :]
        distance = np.linalg.norm(compare - aim)
        distance_list.append(distance)
    # select distances
    smallest_distance = heapq.nsmallest(k, distance_list)
    # get corresponding nodes
    for distance in smallest_distance:
        index = distance_list.index(distance)
        node_list.append(index)
    # for index in range(k):
    #     t_i = node_list[index]
    #     print(stacked_targets[t_i])
    result = Prediction(node_list, targets)
    return result



# read dataset
pwd = os.getcwd()
path = os.path.join(pwd, 'Dataset')
data1_path = os.path.join(path, "X1_t1.csv")
dataset = pandas.read_csv(data1_path)
# extract shape
num_example = dataset.shape[0]
num_feature = dataset.shape[1] - 1
# extract values
dataset = dataset.values
# extract all features
all_features = dataset[:, : -1]
# PCA
reduced_all_features, new_features = PCA_ed(all_features, PCA_k)
np.save("KNN_vector.npy", new_features)
# extract all targets
all_targets = dataset[:, -1]

# # split features
# split_parts = 4
# length = int(num_example / split_parts)
# feature_parts = [reduced_all_features[part * length : (part + 1) * length, :] for part in range(split_parts)]
# stacked_features = np.vstack((feature_parts[0], feature_parts[1], feature_parts[2], feature_parts[3]))
# # split targets
# target_parts = [all_targets[part * length : (part + 1) * length] for part in range(split_parts)]
# stacked_targets = np.hstack((target_parts[0], target_parts[1], target_parts[2], target_parts[3]))

def Ploting(test, test_target, train, target, k):
    num_example = test.shape[0]
    x_axis = [i + 1 for i in range(num_example)]
    prediction_list = []
    for index in range(-num_example, 0):
        aim = test[index, :]
        prediction = KNN(aim, train, target, k)
        prediction_list.append(prediction)
    plt.plot(x_axis, prediction_list,'.b')
    plt.plot(x_axis, test_target,'.r')
    for i in range(num_example):
        x = [i+1, i+1]
        plt.plot(x, [prediction_list[i], test_target[i]], color='k')
    plt.xlabel('Sample')
    plt.ylabel('Output')
    # plt.show()
    error = 0
    for index in range(num_example):
        temp = (prediction_list[index] - test_target[index]) ** 2
        error += temp
        # print(prediction_list[index], test_target[index])
    return math.sqrt(error / num_example)

TNs = [100]
k = 3
for TN in TNs:
    print('k', k, Ploting(reduced_all_features[-TN :, :], all_targets[-TN:], reduced_all_features[: -TN, :], all_targets[: -TN], k))
# for k in range(1, 20):
#     print('k', k, Ploting(reduced_all_features[-TN :, :], all_targets[-TN:], reduced_all_features[: -TN, :], all_targets[: -TN], k))

# # prediction
# data2_path = os.path.join(path, "X2.csv")
# dataset2 = pandas.read_csv(data2_path)
# # extract data
# num_example = dataset2.shape[0]
# num_feature = dataset2.shape[1]
# # extract value
# features = dataset2.values
# # mean
# mean_features = np.array([np.mean(features[:, index]) for index in range(num_feature)])
# norm_features = features - mean_features
# # PCA
# reduced_features = np.matmul(new_features, norm_features.T).T

# prediction_list = []
# for index in range(num_example):
#     aim = reduced_features[index, :]
#     prediction = KNN(aim, reduced_all_features, all_targets, 36)
#     prediction_list.append(prediction)
# csv_prediction = pandas.DataFrame({'prediction': prediction_list})
# csv_prediction.to_csv("X2_KNN.csv", index=False, sep=',')


