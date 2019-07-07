
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau

# https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics


# Rank metrics
################################################################################
ProfA=[3,7,8]
ProfB=[1,5,10]
tau, p_value = kendalltau(ProfA,ProfB)
tau



ProfA=[3,8,7]
ProfB=[1,5,10]
tau, p_value = kendalltau(ProfA,ProfB)
tau


ProfA=[3,7,8]
ProfB=[0,1,1]
tau, p_value = kendalltau(ProfA,ProfB)
tau
2 / np.sqrt((2 + 0 + 0) * (2 + 0 + 1))
# tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
# where P is the number of concordant pairs, Q the number of discordant pairs,
# T the number of ties only in x, and U the number of ties only in y.
# If a tie occurs for the same pair in both x and y, it is not added to either T or U.

2*roc_auc_score(ProfB,ProfA)-1


train = pd.read_csv('./DATA/train.csv')

def miss(ds,n_miss):
	miss_list=list()
	for col in list(ds):
		if ds[col].isna().sum()>=n_miss:
			print(col,ds[col].isna().sum())
			miss_list.append(col)
	return miss_list
# Which columns have 1 missing at least...
print('\n################## TRAIN ##################')
m_tr=miss(train,1)


icol='X30'
print('AUC: ',roc_auc_score(train['TARGET'],train[icol]))
print('GINI: ',2*roc_auc_score(train['TARGET'],train[icol])-1)




# Regression metrics
################################################################################
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

TARGET=[50,55,70,90,87,112,96,75,67,72,84,92,101,98] # People weights in Kg

# horizontal shift
model1=[i+5 for i in TARGET]
mean_absolute_error(TARGET,model)
mean_squared_error(TARGET,model)**0.5


# x2
model2=[i*2 for i in TARGET]
mean_absolute_error(TARGET,model2)
mean_squared_error(TARGET,model2)**0.5


# Outlier
model3=model1.copy()
model3[0]=1000 #
mean_absolute_error(TARGET,model3)
mean_squared_error(TARGET,model3)**0.5
