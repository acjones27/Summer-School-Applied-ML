import numpy as np
import pandas as pd

Feature=[50,55,70,90,87,112,96,75,67,72,84,92,101,98] # People weights in Kg
TARGET=[  0, 0, 1, 1, 0,  1, 1, 0, 0, 0, 0, 1,  1, 1] # TARGET = 1 -> Diabetes


train = pd.DataFrame({'weight':Feature, 'diabetes':TARGET})

# LOGISTIC REGRESSION
# check scikit learn documentation
# penalty = regularisation i.e. lasso (l1), ridge (l2)
# C Inverse of regularization strength --> if we set a C large, we don't regularise --> 1/lambda

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', C=1E20) # Very high C to undo regularization. Normally < 1
# need to put double brackets because only 1 column. expecting matrix not array
model.fit(X = train[['weight']], y = train[['diabetes']])

# B_1
model.coef_
# B_0
model.intercept_

from sklearn.metrics import roc_auc_score
results = train.copy()
# the [:, 1] here is probability of being a 1
# if i put [:, 0], it's the probability of being a 0 i.e a the complementary
results['predictions'] = model.predict_proba(train[['weight']])[:, 1]
auc = roc_auc_score(results['diabetes'], results['predictions'])
gini = 2 * auc - 1
auc
gini

# LOGISTIC REGRESSION MANUAL

results['predictions_Anna'] = 1 / (1 + np.exp( -(model.intercept_[0] + model.coef_[0][0]*train[['weight']]) ))

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(betas, x):
    # Computes the weighted sum of inputs: sum_i(Beta0+Beta1*x_i)
    return np.dot(x, betas)

def probability(betas, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(betas, x))

def cost_function(betas, x, y):
    # Computes the cost function for all the training samples
    # From the video explanation of logitic regression
    total_cost = -np.sum(y*np.log(probability(betas, x))+(1-y)*np.log(1-probability(betas,x))) # Negative we want to minimize
    return total_cost


# Initialize with the feature and TARGET
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

# Transform to arrays in a suitable shape
x = np.c_[np.ones((X.shape[0], 1)), X]   # <---- X=(x1,...,xm) in column to ((1,x1),...,(1,xm)) in column
y = y[:, np.newaxis] # <---- Converts to y=(y1,...,ym) in column as numpy array
betas = np.zeros((x.shape[1], 1)) # <---- initialize betas to betas=(0,0) in column

# MINIMIZE COST FUNCTION
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_tnc.html
from scipy.optimize import fmin_tnc

output=fmin_tnc(func=cost_function, x0=betas,approx_grad=True,args=(x, y.flatten()))
# ----> returns the betas [beta0,beta1]=[intercept,weight] of the minimum of the cost function

beta_vector=output[0]
intercept=beta_vector[0]
weight=beta_vector[1]

# Manually make predictions:
results['predictions_manual']=1/(1+np.exp(-(intercept+weight*results['weight'])))
# Or using defined functions
results['predictions_manual2']=sigmoid(net_input(beta_vector,x))
