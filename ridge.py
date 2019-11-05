# edit this file by replacing ??? with your code

import numpy as np
import matplotlib.pyplot as plt

# LOAD THE DATA AND EDIT IT:
# ==========================
data = np.genfromtxt('housing.data'); # load the data
x = data[:,np.hstack((np.arange(0,3),np.arange(4,9)))] # input data: (a) to (c) end (e) to (i)
(n, d) = x.shape
x = np.hstack((np.ones((x.shape[0],1)), x)); # add 1 for bias term
y = data[:,13] # output data

# CREATE THE TRAIN AND TEST SETS:
# ================================
trainSize = 400; # number of training examples
xTrain = x[np.arange(0,trainSize),:] # training input data
yTrain = y[np.arange(0,trainSize)] # trainint output data
xTest = x[np.arange(trainSize,n),:] # testing input data
yTest = y[np.arange(trainSize,n)] # testing output data

all_lambda = np.exp(np.arange(-5.,10,0.1))
n_lambda = all_lambda.shape[0]
all_trainError = np.zeros(n_lambda)
all_testError = np.zeros(n_lambda)
all_norm = np.zeros(n_lambda)

for i in np.arange(n_lambda):
    lmd = all_lambda[i]
    # COMPUTE LEAST SQUARE ESTIMATE:
    # ==============================
    w_ridge = np.matmul(np.matmul(np.linalg.pinv((lmd*np.eye(d+1)+np.matmul(xTrain.transpose(),xTrain))),xTrain.transpose()),yTrain)
    # TEST THE LINEAR MODEL:
    # ======================
    yPredTrain = np.matmul(xTrain,w_ridge) # generate prediction on training data
    yPredTest = np.matmul(xTest,w_ridge) # generate prediction on testing data
    trainError = np.mean(np.power((yTrain-yPredTrain),2)) # mean-squared error on training data
    testError = np.mean(np.power((yTest-yPredTest),2)) # mean-squared error on test data
    all_trainError[i] = trainError
    all_testError[i] = testError
    all_norm[i] = np.linalg.norm(w_ridge)

plt.rcParams['font.size']=20
plt.semilogx(all_lambda, all_trainError, 'r', all_lambda, all_testError, 'b', all_lambda, all_norm, 'k')
plt.xlabel('lambda')
plt.legend(['TrainError','TestError','Norm'])
plt.show()
