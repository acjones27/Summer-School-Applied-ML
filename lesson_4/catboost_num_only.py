# INITIALIZE
import numpy as np
import pandas as pd
import os
RS=2305 # Seed for partition and model random part
# Read data
print(os.listdir("mesio-summer-school-2019/"))
train = pd.read_csv('mesio-summer-school-2019/train.csv')
test = pd.read_csv('mesio-summer-school-2019/test.csv')

import pandas_profiling as pp
profile = train.profile_report(title = "Training Data Profile Report")
profile.to_file(output_file = "training_data_report.html")

# Warning "A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead" removing:
pd.options.mode.chained_assignment = None

# FEATURE ENGINEERING

# Features types
Features=train.dtypes.reset_index()
Categorical=Features.loc[Features[0]=='object','index']

# Convert categorical to numerical
cols = train.columns.tolist()
cols.remove('TARGET')
test = test[cols]

# X21	[CATEGORICAL] sales (n) / sales (n-1) (buckets of %)
# X43	[CATEGORICAL] rotation receivables + inventory turnover (buckets of days)
# X55	[CATEGORICAL] working capital (capital buckets)

train['X21'].unique()
test['X21'].unique()
# '>100%', '90%-100%', '80%-90%' '75%-80%', '<75%', nan
# let nans be imputed later

train['X43'].unique()
test['X43'].unique()
# 'above 240 days',  '120-240 days', '60-120 days', '30-60 days','less 1 month'

train['X55'].unique()
test['X55'].unique()
# 'above 2000', '500 to 2000', '0 to 500', 'negative'

replace_map = {"X21": {">100%": 150.,
					   "90%-100%": 95.,
					   "80%-90%": 85.,
					   "75%-80%": 77.5,
					   "<75%": 37.5},
			   "X43": {"above 240 days": 300.,
			   		   "120-240 days": 180.,
					   "60-120 days": 90.,
					   "30-60 days": 45.,
					   "less 1 month": 15.},
			   "X55": {"above 2000": 4000.,
			   		   "500 to 2000": 1250.,
					   "0 to 500": 250,
					   "negative": -2000}
			   }

train.replace(replace_map, inplace=True)
test.replace(replace_map, inplace=True)

Features=train.dtypes.reset_index()
Categorical=Features.loc[Features[0]=='object','index']
# 1) Missings
################################################################################
# Function to print columns with at least n_miss missings
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
print('\n################## TEST ##################')
m_te=miss(test,1)

# 1.1) Missings -> Drop some rows
################################################################################
# get observations that have many columns with missing values:

missing_by_row = pd.DataFrame()
missing_by_row['missing_count'] = train.isnull().sum(axis=1).sort_values(ascending = False)

missing_by_row['ID'] = train.iloc[train.isnull().sum(axis=1).sort_values(ascending = False).index.values ]['ID']

# Are all of them in train?
len(train.loc[train['ID']=='A3055',])
len(train.loc[train['ID']=='A1039',])
len(train.loc[train['ID']=='A4665',])
len(train.loc[train['ID']=='A5500',])
len(train.loc[train['ID']=='A1835',])
len(train.loc[train['ID']=='A3160',])
len(train.loc[train['ID']=='A2983',])
len(train.loc[train['ID']=='A3655',])
len(train.loc[train['ID']=='A546',])
len(train.loc[train['ID']=='A1081',])
len(train.loc[train['ID']=='A4415',])
len(train.loc[train['ID']=='A4178',])
len(train.loc[train['ID']=='A1321',])
len(train.loc[train['ID']=='A2029',])
len(train.loc[train['ID']=='A3717',])
# ok, drop:
train = train[train['ID']!='A3055']
train = train[train['ID']!='A1039']
train = train[train['ID']!='A4665']
train = train[train['ID']!='A5500']
train = train[train['ID']!='A1835']
train = train[train['ID']!='A3160']
train = train[train['ID']!='A2983']
train = train[train['ID']!='A3655']
train = train[train['ID']!='A546']
train = train[train['ID']!='A1081']
train = train[train['ID']!='A4415']
train = train[train['ID']!='A4178']
train = train[train['ID']!='A1321']
train = train[train['ID']!='A2029']
train = train[train['ID']!='A3717']

train.reset_index(drop=True,inplace=True)


# 1.2) Missings -> Max or min
################################################################################
# Now, we consider columns with "many" missings:
print('\n################## TRAIN ##################')
m_tr=miss(train,50)

# And plot them to assign maximum or minimum:

# Plot: Features with missing values to impute a value
# Bars = Population in each bucket (left axis)
# Line = Observed Default Frequency (ODF) (right axis)
import matplotlib.pyplot as plt

def feat_graph(df, icol, binary_col, n_buckets):
    feat_data=df[[icol,binary_col]]
    feat_data['bucket']=pd.qcut(feat_data.iloc[:,0], q=n_buckets,labels=False,duplicates='drop')+1

    if len(feat_data.loc[feat_data[icol].isna(),'bucket'])>0:
        feat_data.loc[feat_data[icol].isna(),'bucket']=0

    hist_data_p=pd.DataFrame(feat_data[['bucket',binary_col]].groupby(['bucket'])[binary_col].mean()).reset_index()
    hist_data_N=pd.DataFrame(feat_data[['bucket',binary_col]].groupby(['bucket'])[binary_col].count()).reset_index()
    hist_data=pd.merge(hist_data_N,hist_data_p,how='left',on='bucket')
    hist_data.columns=['bucket','N','p']

    width = .70 # width of a bar
    hist_data['N'].plot(kind='bar', width = width, color='darkgray')
    hist_data['p'].plot(secondary_y=True,marker='o')
    ax = plt.gca()
    plt.xlim([-width, len(hist_data)-width/2])
    if len(feat_data.loc[feat_data[icol].isna(),'bucket'])>0:
        lab=['Missing']
        for i in range(1,n_buckets+1):
            lab.append('G'+str(i))
        ax.set_xticklabels(lab)
    else:
        lab=[]
        for i in range(1,n_buckets+1):
            lab.append('G'+str(i))
        ax.set_xticklabels(lab)

    plt.title(icol)
    plt.show()

for icol in list(m_tr):
    feat_graph(train, icol, 'TARGET', 10)


Impute_min=['X27', 'X28', 'X37', 'X45', 'X53', 'X54', 'X60', 'X64']
Impute_max=['X24', 'X41']

miss_dummy_tr=pd.DataFrame()
miss_dummy_te=pd.DataFrame()

# Create dummies and impute a value far from the minimum (1 standard deviation)
# to be sure that in the test set we are also below the minimum
for col in Impute_min:
	# Dummy colum with the original value
	miss_dummy_tr[col+'_m']=train[col]
	miss_dummy_te[col+'_m']=test[col]

	miss_dummy_tr.loc[~miss_dummy_tr[col+'_m'].isna(),col+'_m']=0 # Not missing -> 0
	miss_dummy_tr.loc[miss_dummy_tr[col+'_m'].isna(),col+'_m']=1 # Missing -> 1
	miss_dummy_te.loc[~miss_dummy_te[col+'_m'].isna(),col+'_m']=0 # Not missing -> 0
	miss_dummy_te.loc[miss_dummy_te[col+'_m'].isna(),col+'_m']=1 # Missing -> 1

	# Fill in train and test with min-std (from train)
	value=train[col].min()-train[col].std()
	train.loc[train[col].isna(),col]=value
	test.loc[test[col].isna(),col]=value

# The same for maximum
for col in Impute_max:
	# Dummy colum with the original value
	miss_dummy_tr[col+'_m']=train[col]
	miss_dummy_te[col+'_m']=test[col]

	miss_dummy_tr.loc[~miss_dummy_tr[col+'_m'].isna(),col+'_m']=0 # No missing -> 0
	miss_dummy_tr.loc[miss_dummy_tr[col+'_m'].isna(),col+'_m']=1 # Missing -> 1
	miss_dummy_te.loc[~miss_dummy_te[col+'_m'].isna(),col+'_m']=0 # No missing -> 0
	miss_dummy_te.loc[miss_dummy_te[col+'_m'].isna(),col+'_m']=1 # Missing -> 1

	# Fill in train and test with max+std (from train)
	value=train[col].max()+train[col].std()
	train.loc[train[col].isna(),col]=value
	test.loc[test[col].isna(),col]=value

# 1.3) Missings -> Exotic techniques
################################################################################
# The remaining missings will be imputed via Iterative Imputer:
# Models each feature with missing values as a function of other features, and
# uses that estimate for imputation

X_train=train.drop(columns=Categorical,axis=1)
X_train.drop(columns='TARGET',axis=1,inplace=True)
X_test=test.drop(columns=Categorical,axis=1)

# Impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

filler=IterativeImputer(n_nearest_features=10, initial_strategy = "median", random_state = RS)
X_train_filled = filler.fit_transform(X_train)
X_test_filled = filler.transform(X_test)

X_train_filled = pd.DataFrame(X_train_filled, columns=list(X_train))
X_test_filled = pd.DataFrame(X_test_filled, columns=list(X_test))

train=pd.concat([train[Categorical],X_train_filled,train['TARGET']],axis=1)
test=pd.concat([test[Categorical],X_test_filled],axis=1)

# Final check:
miss(train,1)
miss(test,1)

# 2) Correlations
################################################################################
# Let's see if certain columns are correlated
# or even that are the same with a "shift"
thresholdCorrelation = 0.99
def InspectCorrelated(df):
	corrMatrix = df.corr().abs() # Correlation Matrix
	upperMatrix = corrMatrix.where(np.triu(np.ones(corrMatrix.shape),k=1).astype(np.bool))
	correlColumns=[]
	for col in upperMatrix.columns:
		correls=upperMatrix.loc[upperMatrix[col]>thresholdCorrelation,col].keys()
		if (len(correls)>=1):
			correlColumns.append(col)
			print("\n",col,'->', end=" ")
			for i in correls:
				print(i, end=" ")
	print('\nSelected columns to drop:\n',correlColumns)
	return(correlColumns)

# Look at correlations in the original features
correlColumns=InspectCorrelated(train.iloc[:,len(Categorical):-1])
# Look at correlations in missing dummies
correlColumns_miss=InspectCorrelated(miss_dummy_tr)

# If we are ok, throw them:
train=train.drop(correlColumns,axis=1)
test=test.drop(correlColumns,axis=1)
miss_dummy_tr=miss_dummy_tr.drop(correlColumns_miss,axis=1)
miss_dummy_te=miss_dummy_te.drop(correlColumns_miss,axis=1)


# 3) Constants
################################################################################
# Let's see if there is some constant column:
def InspectConstant(df):
	consColumns=[]
	for col in list(df):
		if len(df[col].unique())<2:
			print(df[col].dtypes,'\t',col,len(df[col].unique()))
			consColumns.append(col)
	print('\nSelected columns to drop:\n',consColumns)
	return(consColumns)

consColumns=InspectConstant(train.iloc[:,len(Categorical):-1])

# If we are ok, throw them:
train=train.drop(consColumns,axis=1)
test=test.drop(consColumns,axis=1)


# 4) Automatic Alert Creation
################################################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

alerts_train=pd.DataFrame()
alerts_test=pd.DataFrame()
GINIS=list()
ACTIVATIONS=list()
TMRS=list()

for FACTOR in list(train)[len(Categorical):-1]:
	# depth 1 tree
	X=train[[FACTOR]].reset_index(drop=True)
	Y=train['TARGET'].reset_index(drop=True)
	dtree=DecisionTreeClassifier(max_depth=1, random_state=RS)
	dtree.fit(X,Y)
	# Optimal split
	threshold = dtree.tree_.threshold[0]
	# Alert creation
	alerts_train[FACTOR]=train[FACTOR]
	alerts_train[FACTOR+'_b']=np.zeros(len(train))
	alerts_train.loc[alerts_train[FACTOR]<=threshold,FACTOR+'_b']=1

	alerts_test[FACTOR]=test[FACTOR]
	alerts_test[FACTOR+'_b']=np.zeros(len(test))
	alerts_test.loc[alerts_test[FACTOR]<=threshold,FACTOR+'_b']=1

	# Alert in train
	A_train=alerts_train[FACTOR+'_b']

	# GINI of the alert
	gini=roc_auc_score(Y,A_train)*2-1

	# If GINI is negative, we reverse 0 and 1
	if gini<0:
		alerts_train[FACTOR+'_b']=np.logical_not(alerts_train[FACTOR+'_b']).astype(int)
		alerts_test[FACTOR+'_b']=np.logical_not(alerts_test[FACTOR+'_b']).astype(int)

	# GINI again to be sure that all are positive and later save the value to the alert information
	A_train=alerts_train[FACTOR+'_b']
	gini=roc_auc_score(Y,A_train)*2-1

	# ACTIVATIONS
	activ=int(A_train.sum())

	# TMR
	TMO=pd.DataFrame(pd.concat([A_train,Y],axis=1).groupby([FACTOR+'_b'])['TARGET'].mean()).reset_index()
	TMR=float(TMO.loc[TMO[FACTOR+'_b']==1,'TARGET'])/Y.mean()

	# Throw the original factor
	alerts_train.drop([FACTOR],axis=1,inplace=True)
	alerts_test.drop([FACTOR],axis=1,inplace=True)

	# Add GINI, ACTIVATIONS and TMR to the sequence
	GINIS.append(gini*100)
	ACTIVATIONS.append(activ)
	TMRS.append(TMR*100)

# Severities table
severity=pd.DataFrame({'Alert':list(alerts_train),
						'Gini (%)':GINIS,
						'Activations (N)': ACTIVATIONS,
						'TMR (%)': TMRS,
						'TMR/Act': [a/b for a,b in zip(TMRS,ACTIVATIONS)]})

severity=severity.sort_values(by="TMR (%)",ascending=False).reset_index(drop=True)

# Correlations between alerts
# First, we order them by its importance
alerts_train=alerts_train[severity['Alert']]

thresholdCorrelation = 0.95
correlColumns=InspectCorrelated(alerts_train)

# If we are ok, throw them:
alerts_train=alerts_train.drop(correlColumns,axis=1)
alerts_test=alerts_test.drop(correlColumns,axis=1)
for col in correlColumns:
	severity=severity[severity['Alert']!=col].reset_index(drop=True)

# Automatic cut-offs
# We dfine "alert" if it has a minimm TMR of 110
# We split "High", "Medium" and "Low" at thirds
cutoff=round(len(severity.loc[severity['TMR (%)']>=120])/3)
high=severity.loc[0:cutoff-1,'Alert'].tolist()
medium=severity.loc[cutoff:2*cutoff-1,'Alert'].tolist()
low=severity.loc[2*cutoff:3*cutoff-1,'Alert'].tolist()

severity['Severity']=''
severity.loc[severity['Alert'].isin(high),'Severity']='High'
severity.loc[severity['Alert'].isin(medium),'Severity']='Medium'
severity.loc[severity['Alert'].isin(low),'Severity']='Low'

print(severity.to_string())

# Alert counters
# TRAIN
# high
alerts_high_train=alerts_train[high]
alerts_high_train['COUNT_HIGH']=alerts_high_train.sum(axis=1)
# medium
alerts_medium_train=alerts_train[medium]
alerts_medium_train['COUNT_MEDIUM']=alerts_medium_train.sum(axis=1)
# low
alerts_low_train=alerts_train[low]
alerts_low_train['COUNT_LOW']=alerts_low_train.sum(axis=1)
# Añadimos contadores a los datos
train['COUNT_HIGH']=alerts_high_train['COUNT_HIGH']
train['COUNT_MEDIUM']=alerts_medium_train['COUNT_MEDIUM']
train['COUNT_LOW']=alerts_low_train['COUNT_LOW']

# TEST
# high
alerts_high_test=alerts_test[high]
alerts_high_test['COUNT_HIGH']=alerts_high_test.sum(axis=1)
# medium
alerts_medium_test=alerts_test[medium]
alerts_medium_test['COUNT_MEDIUM']=alerts_medium_test.sum(axis=1)
# low
alerts_low_test=alerts_test[low]
alerts_low_test['COUNT_LOW']=alerts_low_test.sum(axis=1)
# Añadimos contadores a los datos
test['COUNT_HIGH']=alerts_high_test['COUNT_HIGH']
test['COUNT_MEDIUM']=alerts_medium_test['COUNT_MEDIUM']
test['COUNT_LOW']=alerts_low_test['COUNT_LOW']


# Finally add missing dummies to datasets
train=pd.concat([train,miss_dummy_tr],axis=1)
test=pd.concat([test,miss_dummy_te],axis=1)


# 5) WoE
################################################################################
# Woe Function
def WoE(icol,binary_col,df_train,df_test,n_buckets=None):
	if n_buckets: # If the feature is continuous
		df_train['bucket'], bins = pd.qcut(df_train[icol],q=n_buckets,labels=False,duplicates='drop',retbins=True)
		real_bins=len(bins)-1
		df_test['bucket'] = pd.cut(df_test[icol],bins=bins,labels=False,include_lowest=True)
		# If we are below the minimum or above the maximum in test assign the extreme buckets:
		df_test.loc[(df_test['bucket'].isna()) & (df_test[icol]>=max(bins)),'bucket']=real_bins-1
		df_test.loc[(df_test['bucket'].isna()) & (df_test[icol]<=min(bins)),'bucket']=0
		woe_table=df_train[['bucket',binary_col]].groupby(['bucket']).sum(skipna=True).reset_index()
	else: # If we do not specify n_buckets, assume that the feature is categorical or discrete (treated the same way)
		df_train['bucket']=df_train[icol]
		df_test['bucket']=df_test[icol]
		real_bins=len(df_train[icol].unique())
		woe_table=df_train[[icol,binary_col]].groupby([icol]).sum(skipna=True).reset_index()
		woe_table = woe_table.rename(columns={icol: 'bucket'})

	# GOOD & BAD Total
	BAD=df_train[binary_col].sum(skipna=True)
	GOOD=df_train.loc[~df_train[binary_col].isna(),binary_col].count()-BAD

	# We have at least 2 values
	if real_bins>=2:
		woe_table = woe_table.rename(columns={binary_col: 'BAD'}) # Defaults
		woe_table['TOTAL']=df_train[['bucket',binary_col]].groupby(['bucket']).count().reset_index()[binary_col] # Totales
		woe_table['GOOD']=(woe_table['TOTAL']-woe_table['BAD']).astype(int) # Buenos

		# WoE by bucket
		woe_table['WOE']=np.log(((woe_table['GOOD']+0.5)/GOOD)/((woe_table['BAD']+0.5)/BAD))

		# Add the new factor and remove the original
		df_train = pd.merge(df_train, woe_table[['bucket','WOE']], on='bucket', how='left')
		df_train = df_train.rename(columns={'WOE': icol+"_W"})
		df_train = df_train.drop(icol, axis=1)
		df_train = df_train.drop('bucket', axis=1)

		df_test = pd.merge(df_test, woe_table[['bucket','WOE']], on='bucket', how='left')
		# In case that for a Categorical variable (for Numerical variables this
		# is impossible since we have assigned every observation to a bin)
		# there are unseen categories in test (not found in train)
		# -> assign WoE = 0 (neutral WoE)
		df_test.loc[df_test['WOE'].isna(),'WOE']=0
		df_test = df_test.rename(columns={'WOE': icol+"_W"})
		df_test = df_test.drop(icol, axis=1)
		df_test = df_test.drop('bucket', axis=1)
	else:
		print('Column ',icol,' has less than 2 buckets -> Removed')
		df_train = df_train.drop(icol, axis=1)
		df_train = df_train.drop('bucket', axis=1)
		df_test = df_test.drop(icol, axis=1)
		df_test = df_test.drop('bucket', axis=1)

	return df_train, df_test


# List of features that we will treat as Categorical for WoE
As_Categorical=Categorical.tolist()
As_Categorical.remove('ID')
As_Categorical.append('COUNT_HIGH')
As_Categorical.append('COUNT_MEDIUM')
As_Categorical.append('COUNT_LOW')
miss_dummies = [i for i in list(train) if '_m' in i]
for i in miss_dummies:
	As_Categorical.append(i)

# List of features that we will treat as Numerical for WoE
As_Numerical=list(train)
As_Numerical.remove('ID')
As_Numerical.remove('TARGET')
for i in As_Categorical:
	As_Numerical.remove(i)

# Initialize woe (or lineal) sets for modeling
train_woe=train.copy()
test_woe=test.copy()

# Transform Categorical
for icol in As_Categorical:
	train_woe, test_woe = WoE(icol=icol,
							  binary_col='TARGET',
							  df_train=train_woe,
							  df_test=test_woe,
							  n_buckets=None)
# Transform Numcerical
for icol in As_Numerical:
	train_woe, test_woe = WoE(icol=icol,
							  binary_col='TARGET',
							  df_train=train_woe,
							  df_test=test_woe,
							  n_buckets=10)

# Reorder columns (TRAIN at the end)
cols=list(train)
cols.insert(len(cols), cols.pop(cols.index('TARGET')))
train = train.reindex(columns= cols)

cols=list(train_woe)
cols.insert(len(cols), cols.pop(cols.index('TARGET')))
train_woe = train_woe.reindex(columns= cols)

# MODELING

# Train and test (validation) sets for all kind of models
################################################################################

# For non-linear models
predictoras=list(train)[1:-1]
X_train=train[predictoras].reset_index(drop=True)
Y_train=train['TARGET'].reset_index(drop=True)
X_test=test[predictoras].reset_index(drop=True)

# For linear models
predictoras_woe=list(train_woe)[1:-1]
X_train_woe=train_woe[predictoras_woe].reset_index(drop=True)
X_test_woe=test_woe[predictoras_woe].reset_index(drop=True)


# MODEL CATBOOST

# 1) For expensive models (catboost) we first try with validation set (no cv)
################################################################################
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn import preprocessing
# train / test partition
TS=0.3 # Validation size
esr=50 # Early stopping rounds (when validation does not improve in these rounds, stops)

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=TS, random_state=RS)
#
Features=x_tr.dtypes.reset_index()
Categorical=Features.loc[Features[0]=='object','index']

tr=x_tr.drop(columns=Categorical,axis=1)
val=x_val.drop(columns=Categorical,axis=1)

scaler = preprocessing.StandardScaler()
tr = pd.DataFrame(scaler.fit_transform(tr), columns = tr.columns).reset_index(drop = True)
val = pd.DataFrame(scaler.transform(val), columns = tr.columns).reset_index(drop = True)

tr_cat = x_tr[Categorical].reset_index(drop = True)
val_cat = x_val[Categorical].reset_index(drop = True)

x_tr=pd.concat([tr_cat,tr],axis=1)
x_val=pd.concat([val_cat,val],axis=1)


# Categorical positions for catboost
Pos=list()
As_Categorical=Categorical.tolist()
# As_Categorical.remove('ID')
for col in As_Categorical:
    Pos.append((X_train.columns.get_loc(col)))

# To Pool Class (for catboost only)
pool_tr=Pool(x_tr, y_tr,cat_features=Pos)
pool_val=Pool(x_val, y_val,cat_features=Pos)

# By-hand paramter tuning. A grid-search is expensive
# We test different combinations
# See parameter options here:
# "https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/"
model_catboost_val = CatBoostClassifier(
          eval_metric='AUC',
          iterations=20000, # Very high value, to find the optimum
          od_type='Iter', # Overfitting detector set to "iterations" or number of trees
          random_seed=RS, # Random seed for reproducibility
          verbose=100) # Shows train/test metric every "verbose" trees

# "Technical" parameters of the model:
params = {'objective': 'Logloss',
		  'learning_rate': 0.005, # learning rate, lower -> slower but better prediction
		  'depth': 5, # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
		  'l2_leaf_reg': 10, # L2 regularization (between 3 and 20, higher -> less overfitting)
		  'rsm': 0.4, # % of features to consider in each split (lower -> faster and reduces overfitting)
		  'bootstrap_type': 'Bayesian'} # For categorical variables

model_catboost_val.set_params(**params)

print('\nCatboost Fit (Validation)...\n')
model_catboost_val.fit(X=pool_tr,
                       eval_set=pool_val,
                       early_stopping_rounds=esr)

# 2) Cross-Validation (Catboost)
################################################################################

# 2.1) k-Fold Cross-Validation Function
################################################################################
from sklearn.model_selection import StratifiedKFold

def Model_cv(MODEL, k, X_train, X_test, y, RE, makepred=True, CatPos=None):
	# Create the k folds
	kf=StratifiedKFold(n_splits=k, shuffle=True, random_state=RE)

	# first level train and test
	Level_1_train = pd.DataFrame(np.zeros((X_train.shape[0],1)), columns=['train_yhat'])
	if makepred==True:
		Level_1_test = pd.DataFrame()

	# Main loop for each fold. Initialize counter
	count=0
	for train_index, test_index in kf.split(X_train, Y_train):
		count+=1

		Features=X_train.dtypes.reset_index()
		Categorical=Features.loc[Features[0]=='object','index']

		tr=X_train.drop(columns=Categorical,axis=1)
		val=X_test.drop(columns=Categorical,axis=1)

		scaler = preprocessing.StandardScaler()
		tr = pd.DataFrame(scaler.fit_transform(tr), columns = tr.columns).reset_index(drop = True)
		val = pd.DataFrame(scaler.transform(val), columns = tr.columns).reset_index(drop = True)

		tr_cat = X_train[Categorical].reset_index(drop = True)
		val_cat = X_test[Categorical].reset_index(drop = True)

		X_train=pd.concat([tr_cat,tr],axis=1)
		X_test=pd.concat([val_cat,val],axis=1)

		# Define train and test depending in which fold are we
		fold_train= X_train.loc[train_index.tolist(), :]
		fold_test=X_train.loc[test_index.tolist(), :]
		fold_ytrain=y[train_index.tolist()]
		fold_ytest=y[test_index.tolist()]

		# (k-1)-folds model adjusting
		if CatPos:
			# Prepare Pool
			pool_train=Pool(fold_train, fold_ytrain,cat_features=Pos)
			# (k-1)-folds model adjusting
			model_fit=MODEL.fit(X=pool_train)

		else:
			# (k-1)-folds model adjusting
			model_fit=MODEL.fit(fold_train, fold_ytrain)

		# Predict on the free fold to evaluate metric
		# and on train to have an overfitting-free prediction for the next level
		p_fold=MODEL.predict_proba(fold_test)[:,1]
		p_fold_train=MODEL.predict_proba(fold_train)[:,1]

		# Score in the free fold
		score=roc_auc_score(fold_ytest,p_fold)
		score_train=roc_auc_score(fold_ytrain,p_fold_train)
		print(k, '-cv, Fold ', count, '\t --test AUC: ', round(score,4), '\t--train AUC: ', round(score_train,4),sep='')
		# Save in Level_1_train the "free" predictions concatenated
		Level_1_train.loc[test_index.tolist(),'train_yhat'] = p_fold

		# Predict in test to make the k model mean
		# Define name of the prediction (p_"iteration number")
		if makepred==True:
			name = 'p_' + str(count)
			# Predictin to real test
			real_pred = MODEL.predict_proba(X_test)[:,1]
			# Name
			real_pred = pd.DataFrame({name:real_pred}, columns=[name])
			# Add to Level_1_test
			Level_1_test=pd.concat((Level_1_test,real_pred),axis=1)

	# Compute the metric of the total concatenated prediction (and free of overfitting) in train
	score_total=roc_auc_score(y,Level_1_train['train_yhat'])
	print('\n',k, '- cv, TOTAL AUC:', round((score_total)*100,4),'%')

	# mean of the k predictions in test
	if makepred==True:
		Level_1_test['model']=Level_1_test.mean(axis=1)

	# Return train and test sets with predictions and the performance
	if makepred==True:
		return Level_1_train, pd.DataFrame({'test_yhat':Level_1_test['model']}), score_total
	else:
		return score_total

# 2.2) k-Fold Cross-Validation Implementattion
################################################################################
# Parameters of the CV
n_folds=5 # Number of folds (depends on the sample size, the proportion of 1's over 0's,...)

# Define the model
model_catboost_cv=CatBoostClassifier()
model_catboost_cv.set_params(**params)
model_catboost_cv.set_params(random_seed=RS)
model_catboost_cv.set_params(verbose=False)

# Put in the "iter" list various values around the discovered in the previous step:
# (The number of iterations is altered proportionaly in function of the
# datasets sizes (where has been obtained and where has to be applied))
nrounds_cv=round(model_catboost_val.best_iteration_/(1-TS)*(1-1/n_folds))
iter=[round(nrounds_cv*0.9),nrounds_cv,round(nrounds_cv*1.1)]

print('\nCatboost CV...')
print('########################################################')
scores=[]
for nrounds in iter:
	model_catboost_cv.set_params(n_estimators=nrounds)
	print('\nn rounds: ',nrounds)
	Pred_train, Pred_test, s = Model_cv(model_catboost_cv,n_folds,X_train,X_test,Y_train,RS,makepred=True,CatPos=Pos)

	# Look if we are in the first test:
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	# If the score improves, we keep this one:
	if s>=max_score:
		print('BEST')
		Catboost_train=Pred_train.copy()
		Catboost_test=Pred_test.copy()

	# Append score
	scores.append(s)

# The best cross-validated score has been found in:
print('\n###########################################')
print('Catboost optimal rounds: ',iter[scores.index(max(scores))])
print('Catboost optimal GINI: ',round((max(scores)*2-1)*100,4),'%')
print('Catboost optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')


# 2.3) Train a model on all the whole train with the optimal parameters:
################################################################################

# Adjust optimal CV number of rounds to whole sample size:
nrounds=int(iter[scores.index(max(scores))]/(1-1/n_folds))

# Define the optimal model
model_catboost=CatBoostClassifier(n_estimators=nrounds,
								  random_seed=RS,
								  verbose=100)
model_catboost.set_params(**params)

Features=X_train.dtypes.reset_index()
Categorical=Features.loc[Features[0]=='object','index']

tr=X_train.drop(columns=Categorical,axis=1)
val=X_test.drop(columns=Categorical,axis=1)

scaler = preprocessing.StandardScaler()
tr = pd.DataFrame(scaler.fit_transform(tr), columns = tr.columns).reset_index(drop = True)
val = pd.DataFrame(scaler.transform(val), columns = tr.columns).reset_index(drop = True)

tr_cat = X_train[Categorical].reset_index(drop = True)
val_cat = X_test[Categorical].reset_index(drop = True)

X_train=pd.concat([tr_cat,tr],axis=1)
X_test=pd.concat([val_cat,val],axis=1)

# To Pool Class (for catboost only)
pool_train=Pool(X_train, Y_train,cat_features=Pos)

# Fit the model
print('\nCatboost Optimal Fit with %d rounds...\n' % nrounds)
model_catboost.fit(X=pool_train)

# RESULTS

# Prediction
test['Pred']=model_catboost.predict_proba(X_test)[:,1]
outputs_catboost=pd.DataFrame(test[['ID','Pred']])

# Outputs to .csv
outputs_catboost.to_csv("outputs_catboost_scaled_convertcatnum_param_tweak.csv", index = False)
