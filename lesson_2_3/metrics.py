import pandas as pd # working with dataframes
import numpy as np # working with arrays
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau

# Rank Metrics
ProfA = [3, 7, 8]
ProfB = [1, 5, 10]
tau, p_value = kendalltau(ProfA, ProfB)
tau
# (1 + 1 + 1)/3 = 100%

ProfA = [3, 8, 7]
ProfB = [1, 5, 10]
tau, p_value = kendalltau(ProfA, ProfB)
tau
# (1 + 1 - 1)/3 = 100%

ProfA = [3, 8, 7]
ProfB = [0, 1, 1]
tau, p_value = kendalltau(ProfA, ProfB)
tau
# There's a clear point to separate to have all the 1s on one side and all the 0s on the other so it;s a perfect  model
# (1 + 1)/2 = 100%, ignore the pair where ties

# tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
# where P is the number of concordant pairs

2 * roc_auc_score(ProfB, ProfB) - 1
# only accepted when ProfB is 0/1

train = pd.read_csv('mesio-summer-school-2019/train.csv', )
train.head()

# col names
train.columns

# col names
list(train)

# test.csv doesn't have the TARGET column. That's in the kaggle comp

# call one column
train['X1']

# return a list of columns with at least n_miss missings
def miss(ds, n_miss):
    # define empty list
    miss_list = list()
    # loop over list of column names
    for col in list(ds):
        # if the column has more than n_miss missings
        if ds[col].isna().sum() >= n_miss:
            # print the name of the column and how many missings
            print(col, ds[col].isna().sum())
            # append to miss_list the name of the column with missings
            miss_list.append(col)
    return miss_list

print("TRAIN")

# return list of columns with at least n missings
# it will print the column name and how many are missing
m_tr = miss(train, 30)

# select col with no missings
col = 'X30'
train[col]
print("AUC: ", roc_auc_score(train['TARGET'], train[col]))
print("GINI: ", 2*roc_auc_score(train['TARGET'], train[col]) - 1)
# AUC = 0.68
# GINI = 0.37
# If i only select 1 column, column X30, i would only get a gini of 37%

# Regression Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

TARGET = [50, 55, 70, 90, 87, 112, 96, 75, 67, 72, 84, 92, 101, 98]

# horizontal shift
model1 = [i+5 for i in TARGET]
mean_absolute_error(TARGET, model1)
mean_squared_error(TARGET, model1)**0.5
len(TARGET)

# x2
model2 = [i*2 for i in TARGET]
mean_absolute_error(TARGET, model2)
mean_squared_error(TARGET, model2)**0.5

# outlier
model3 = model1.copy()
model3[0] = 1000
mean_absolute_error(TARGET, model3)
mean_squared_error(TARGET, model3)**0.5
# mean squared error is more sensitive to outliers
# MAE is more robust to outliers
