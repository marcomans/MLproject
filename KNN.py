import numpy as np
import pandas
import os
import heapq
from PCA import PCA_ed


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
    print('k', k, '+', result)



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
reduced_all_features = PCA_ed(all_features, PCA_k)
# extract all targets
all_targets = dataset[:, -1]

# split features
split_parts = 4
length = int(num_example / split_parts)
feature_parts = [reduced_all_features[part * length : (part + 1) * length, :] for part in range(split_parts)]
stacked_features = np.vstack((feature_parts[0], feature_parts[1], feature_parts[2]))
# print(stacked_features.shape)
# split targets
target_parts = [all_targets[part * length : (part + 1) * length] for part in range(split_parts)]
stacked_targets = np.hstack((target_parts[0], target_parts[1], target_parts[2]))
# print(stacked_targets.shape)


for i in range(1, 5):
    aim = reduced_all_features[-i, :]
    print('aim: ', dataset[-i, -1])
    for k in range(1, 10):
        KNN(aim, stacked_features, stacked_targets, k)
