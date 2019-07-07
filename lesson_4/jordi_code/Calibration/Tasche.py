##########################################################################################################
###########################################             ##################################################
########################################### TASCHE CODE ##################################################
###########################################             ##################################################
##########################################################################################################
import numpy as np
from statsmodels import distributions
from scipy.optimize import fsolve
from scipy import stats
from operator import sub
import math

class LDPortfolio:
	'''
	Basic functionality for all LDP calibration facilities (LD=Low Default).
	:attribute self.ar: Estimated Accuracy ratio given portfolio distribution and PD values.
	:attribute self.ct: Central Tendency (mean PD) ratio given portfolio distribution and PD values.
	'''
	def __init__(self, portfolio, pd_cnd=None):
		'''
		:param portfolio: Unconditional portfolio distribution from the worst to the best credit quality.
						Each 'portfolio' item contains total number of observations in a given rating class.
		:param pd_cnd: Current conditional PD distribution from the worst to the best credit quality.
						Used for current AR estimation.
		'''
		self.portfolio = np.array(portfolio)
		self.pd_cnd = np.array(pd_cnd)
		self.portfolio_size = self.portfolio.sum()
		# self.portfolio_dist = self.portfolio.cumsum() / self.portfolio_size
		# self.portfolio_dist = (np.hstack((0, self.portfolio_dist[:-1])) + self.portfolio_dist) / 2
		self.rating_prob = self.portfolio / self.portfolio_size

		self.ct = None
		self.ar = None
		if not pd_cnd is None:
			self.ct, self.ar = self._ar_estimate(self.pd_cnd)

	def _ar_estimate(self, pd_cnd):
		ct = self.rating_prob.T.dot(pd_cnd)
		ar_1int_1 = self.rating_prob * pd_cnd
		ar_1int_1 = np.hstack((0, ar_1int_1[:-1]))
		ar_1int_1 =  (1 - pd_cnd) * self.rating_prob * ar_1int_1.cumsum()
		ar_1 = 2 * ar_1int_1.sum()
		ar_2 =  (1 - pd_cnd) * pd_cnd * self.rating_prob * self.rating_prob
		ar = (ar_1 + ar_2.sum()) * (1.0 / (ct * (1 - ct))) - 1
		return ct, ar.sum()

# Quasi Moment Matching implementation
class QMM(LDPortfolio):
	"""
	Calibrates conditional probabilities of default according to Quasi Moment Matching algorithm
	:attribute self.pd_cnd: calibrated conditional PD
	:attribute self.alpha: intercept calibration parameter
	:attribute self.beta: slope calibration parameter
	"""
	def __init__(self, portfolio, portfolio_cnd_no_dft = None):
		"""
		:param portfolio: Unconditional portfolio distribution from the worst to the best credit quality.
						Each 'portfolio' item contains total number of observations in a given rating class.
		:param portfolio_cnd_no_dft: Conditional on no default portfolio distribution (in case None, unconditional
						portfolio distribution is used as a proxy)
		:return: initialized QMM class object
		"""
		super().__init__(portfolio, pd_cnd=None)
		if portfolio_cnd_no_dft is None:
			self.portfolio_cnd_no_dft = self.portfolio
			self.cum_cond_ND = (np.hstack((0, self.portfolio_cnd_no_dft.cumsum()[:-1])) + self.portfolio_cnd_no_dft.cumsum()) / 2
		else:
			self.cum_cond_ND = (np.hstack((0, portfolio_cnd_no_dft.cumsum()[:-1])) + portfolio_cnd_no_dft.cumsum()) / 2

		self.alpha = None
		self.beta = None

	def fit(self, ct_target, ar_target):
		"""
		:param ct_target: target Central Tendency
		:param ar_target: target Accuracy Ratio
		:return: calibrated QMM class
		"""
		a = self.__get_pd((0, 0))
		tf = lambda x: tuple(map(sub, self._ar_estimate(self.__get_pd(x)), (ct_target, ar_target)))
		params = fsolve(tf, (0, 0))
		self.alpha, self.beta = params
		self.pd_cnd = self.__get_pd(params)
		self.ct, self.ar = self._ar_estimate(self.pd_cnd)
		return self

	def __get_pd(self, params):
		return self._robust_logit(self.cum_cond_ND, params)

	@staticmethod
	def _robust_logit(x, params):
		alpha, beta = params
		return 1 / (1 + np.exp(- alpha - beta * stats.norm.ppf(x)))



##########################################################################################################
#############################################         ####################################################
############################################# EXAMPLE ####################################################
#############################################         ####################################################
##########################################################################################################

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

# Create buckets of score and save it to apply to test
train['bucket'], bins=pd.qcut(train['Pred'], q=n_buckets,labels=False,duplicates='drop',retbins=True)
real_bins=len(bins)-1
train['bucket']=real_bins-train['bucket']

# Calibration table
rating_data_N=pd.DataFrame(train[['bucket','TARGET']].groupby(['bucket'])['TARGET'].count()).reset_index() # Totales
rating_data_D=pd.DataFrame(train[['bucket','TARGET']].groupby(['bucket'])['TARGET'].sum()).reset_index() # Defaults
rating_data=pd.merge(rating_data_N,rating_data_D,how='left',on='bucket')
rating_data.columns=['bucket','N','D']
rating_data['ND']=rating_data['N']-rating_data['D'] # Goods
rating_data['COND_PD']=rating_data['D']/rating_data['N'] # Conditional PD
total_ND=rating_data['ND'].sum() # Total GOODS
rating_data['COND_ND']=rating_data['ND']/total_ND # GOODS distribution


# Tasche model
p1=LDPortfolio(portfolio=rating_data['N'],pd_cnd=rating_data['COND_PD'])
q1=QMM(portfolio=rating_data['N'], portfolio_cnd_no_dft=rating_data['COND_ND'])

train['TARGET'].mean()

# We choose if we change the central tendency
###############
Central_Tendency=p1.ct
# Central_Tendency=0.1
###############

q1.fit(ct_target=Central_Tendency, ar_target=p1.ar)
rating_data['CALIB']=q1.pd_cnd # Nueva curva suavizada


# Plot calibration curve
################################################################################
import matplotlib.pyplot as plt
# Create labels
lab=[]
for i in range(1,real_bins+1):
	lab.append(str(i))
rating_data['labels']=lab
# Plot:
fig=plt.figure(figsize=(24,16))
plt.plot(rating_data['labels'],rating_data['COND_PD'],color='red')
plt.plot(rating_data['labels'],rating_data['CALIB'],color='darkblue',marker='o')
plt.title('Tasche Calibration Curve')
plt.show()






# Plot train & test adjusts
################################################################################

# Application to the whole sample:
################################################################################
train_calib=pd.merge(train,rating_data[['bucket','CALIB']],on='bucket', how='left')

# Application to the test set:
################################################################################
test['bucket'] = pd.cut(test['Pred'], bins=bins, labels=False)
test['bucket']=real_bins-test['bucket']
test_calib=pd.merge(test,rating_data[['bucket','CALIB']],on='bucket', how='left')

# Check central tendencies
train_calib['CALIB'].mean()
test_calib['CALIB'].mean()






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
train_calib['RATING']=(N_class-pd.cut(train_calib['CALIB'], bins=points, labels=False))/2
test_calib['RATING']=(N_class-pd.cut(test_calib['CALIB'], bins=points, labels=False))/2





# train plot
train_calib_data_TMO=pd.DataFrame(train_calib[['RATING','TARGET']].groupby(['RATING'])['TARGET'].mean()).reset_index()
train_calib_data_PD=pd.DataFrame(train_calib[['RATING','CALIB']].groupby(['RATING'])['CALIB'].mean()).reset_index()
train_calib_data_N=pd.DataFrame(train_calib[['RATING','TARGET']].groupby(['RATING'])['TARGET'].count()).reset_index()
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
test_calib_data_PD=pd.DataFrame(test_calib[['RATING','CALIB']].groupby(['RATING'])['CALIB'].mean()).reset_index()
test_calib_data_N=pd.DataFrame(test_calib[['RATING','CALIB']].groupby(['RATING'])['CALIB'].count()).reset_index()
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
