import numpy as np
import os
import pandas


new_features = np.load("linear_vector.npy")
Weight = np.load("Weight6_6.npy")

# read dataset
pwd = os.getcwd()
path = os.path.join(pwd, 'Dataset')
data2_path = os.path.join(path, "X2.csv")
dataset = pandas.read_csv(data2_path)
# extract data
num_example = dataset.shape[0]
num_feature = dataset.shape[1]
# extract value
features = dataset.values
# mean
mean_features = np.array([np.mean(features[:, index]) for index in range(num_feature)])
norm_features = features - mean_features
# PCA
reduced_features = np.matmul(new_features, norm_features.T).T
# bias
bias = np.ones((num_example, 1))
biased_reduced_all_features = np.hstack((bias, reduced_features))

prediction_list = []
for index in range(num_example):
    prediction_list.append(np.dot(biased_reduced_all_features[index, :], Weight))
csv_prediction = pandas.DataFrame({'prediction': prediction_list})
csv_prediction.to_csv("X2_prediction_1.csv", index=False, sep=',')