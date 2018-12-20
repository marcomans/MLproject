from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



X1=pd.read_csv("data/X1_t1.csv")

dataset = X1.values

####
# Feature Selection NEEDED!
####

# features1 = dataset[:,0:2].copy()
# features2 = dataset[:,3:8].copy()
# features = np.concatenate((features1,features2), axis=1)
features = dataset[:,0:8].copy()


target = X1['ccs'].values

####
#Dividing the dataset (train+test) (80%+20%)
pivot = int(0.8*len(dataset))
train = features[:pivot,:]
test = features[pivot:,:]

targetTrain = target[:pivot]
targetTest = target[pivot:]


# create a Linear Regressor
lin_regressor = LinearRegression()

# pass the order of your polynomial here
poly = PolynomialFeatures(2)


X_transform = poly.fit_transform(train)

lin_regressor.fit(X_transform,targetTrain)

# get the predictions
X_test_transf = poly.fit_transform(test)
y_preds = lin_regressor.predict(X_test_transf)
# print(y_preds)
# print(targetTest)

x = np.arange(targetTest.size)
plt.figure()
plt.plot(np.concatenate((np.vstack(x),np.vstack(x)),axis=1).T,
        np.concatenate((np.vstack(targetTest),np.vstack(y_preds)),axis=1).T,'-k')
plt.plot(x,targetTest,'.r')
plt.plot(x,y_preds,'.b')
plt.xlabel('Sample')
plt.ylabel('Output')
plt.title("Polynomial of degree 2")
plt.show()


print("----------\nValues predicted:\n----------\n{}\n----------\n".format(y_preds))
print("----------\nActual value:\n----------\n{}\n----------\n".format(targetTest))
err = np.sqrt(np.mean((y_preds-targetTest)**2))

print("----------\nresidual standard error : {}\n----------\n".format(err))