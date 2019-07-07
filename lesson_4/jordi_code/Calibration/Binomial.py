import numpy as np
import pandas as pd
##########################################################################################################
# SCORE (CONTINUOUS) DATA
##########################################################################################################
train = pd.read_csv('./Model_train.csv',sep=',')
test = pd.read_csv('./Model_test.csv',sep=',')

# Number of calibration buckets (not too small, not too large)
###############
n_buckets=100
###############

# train buckets
################################################################################
train['bucket'], bins=pd.qcut(train['Pred'], q=n_buckets,labels=False,duplicates='drop',retbins=True)
train['bucket']+=1

real_bins=len(bins)-1

# Calibration Table
################################################################################
rating_data_N=pd.DataFrame(train[['bucket','TARGET']].groupby(['bucket'])['TARGET'].count()).reset_index() # Totales
rating_data_D=pd.DataFrame(train[['bucket','TARGET']].groupby(['bucket'])['TARGET'].sum()).reset_index() # Defaults
rating_data=pd.merge(rating_data_N,rating_data_D,how='left',on='bucket')
rating_data.columns=['bucket','N','D']
rating_data['ND']=rating_data['N']-rating_data['D'] # Buenos
rating_data['Real']=rating_data['D']/rating_data['N']

# Mean of predicted probabilities per bucket
rating_data['predicted']=(train[['bucket','Pred']].groupby(['bucket'])['Pred'].mean()).reset_index(drop=True)
# -Log odds
rating_data['score']=-np.log(rating_data['predicted']/(1-rating_data['predicted']))


# Binomial model
################################################################################
import statsmodels.api as sm
# Binomial response
response=np.asarray(pd.concat([rating_data['D'], rating_data['ND']], axis=1))
# Fit de model
glm_binom = sm.GLM(response,sm.add_constant(rating_data['score']), family=sm.families.Binomial()).fit()
# Model parameters
print(glm_binom.summary())

# prediction on the rating_table
rating_data['p_calib']=glm_binom.predict(exog=sm.add_constant(rating_data['score']))


# Plot calibration curve
################################################################################
import matplotlib.pyplot as plt
# Create labels
lab=[]
for i in range(1,real_bins+1):
	lab.append(str(i))
rating_data['labels']=lab
rating_data=rating_data.sort_values(by=['bucket'],ascending=False).reset_index(drop=True)
# Plot:
fig=plt.figure(figsize=(24,16))
plt.plot(rating_data['labels'],rating_data['Real'],color='red')
plt.plot(rating_data['labels'],rating_data['p_calib'],color='darkblue',marker='o')
plt.title('Binomial Calibration Curve')
plt.show()




# Plot train & test adjusts
################################################################################

# Application to the whole sample:
################################################################################
# Obtain the oscore as -log odds
train['score']=-np.log(train['Pred']/(1-train['Pred']))

# Apply calibration model
train['p_calib']=glm_binom.predict(exog=sm.add_constant(train['score']))


# Application to the test set:
################################################################################
# Obtain the oscore as -log odds
test['score']=-np.log(test['Pred']/(1-test['Pred']))

# Apply calibration model
test['p_calib']=glm_binom.predict(exog=sm.add_constant(test['score']))



# Master ladder
x_0=0.0001 # (0.01%)
ladder=list()
while x_0<1:
    ladder.append(x_0)
    x_0=1.5*x_0

N_class=len(ladder)
points=ladder
points.insert(0,0)
points.append(100)




# RATING assignment
train['RATING']=(N_class-pd.cut(train['p_calib'], bins=points, labels=False))/2
test['RATING']=(N_class-pd.cut(test['p_calib'], bins=points, labels=False))/2





# train plot
train_calib_data_TMO=pd.DataFrame(train[['RATING','TARGET']].groupby(['RATING'])['TARGET'].mean()).reset_index()
train_calib_data_PD=pd.DataFrame(train[['RATING','p_calib']].groupby(['RATING'])['p_calib'].mean()).reset_index()
train_calib_data_N=pd.DataFrame(train[['RATING','TARGET']].groupby(['RATING'])['TARGET'].count()).reset_index()
train_calib_data=pd.merge(train_calib_data_N,train_calib_data_TMO,how='left',on='RATING')
train_calib_data=pd.merge(train_calib_data,train_calib_data_PD,how='left',on='RATING')
train_calib_data.columns=['RATING','N','TMO','PD']

fig=plt.figure(figsize=(12,8))
width = .70 # width of a bar
train_calib_data['N'].plot(kind='bar', width = width, color='darkgray',label='N')
train_calib_data['TMO'].plot(secondary_y=True,marker='o',label='TMO',color='orange')
train_calib_data['PD'].plot(secondary_y=True,marker='o',label='PD', color='darkblue')
ax = plt.gca()
plt.xlim([-width, len(train_calib_data)-width/2])

lab=train_calib_data['RATING'].unique()
ax.set_xticklabels(lab)
ax.legend()
plt.title('Train Calibration')
plt.show()






# test plot
test_calib_data_PD=pd.DataFrame(test[['RATING','p_calib']].groupby(['RATING'])['p_calib'].mean()).reset_index()
test_calib_data_N=pd.DataFrame(test[['RATING','p_calib']].groupby(['RATING'])['p_calib'].count()).reset_index()
test_calib_data=pd.merge(test_calib_data_N,test_calib_data_PD,how='left',on='RATING')
test_calib_data.columns=['RATING','N','PD']

fig=plt.figure(figsize=(12,8))
width = .70 # width of a bar
test_calib_data['N'].plot(kind='bar', width = width, color='darkgray',label='N')
test_calib_data['PD'].plot(secondary_y=True,marker='o',label='PD', color='darkblue')
ax = plt.gca()
plt.xlim([-width, len(test_calib_data)-width/2])

lab=test_calib_data['RATING'].unique()
ax.set_xticklabels(lab)
ax.legend()
plt.title('Test Calibration')
plt.show()
