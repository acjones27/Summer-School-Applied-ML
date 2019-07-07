# Lesson 1

## Kaggle Competition

[Link](https://www.kaggle.com/c/mesio-summer-school-2019/overview)

### Spanish vocab
- imputar missings --> fill missing data
- foros --> forums
- escargar --> to order, to commission, to entrust or put in charge of
- etiqueta --> a tag or label
- conceder --> to grant or admit
- captar --> to capture or understand
- incremento --> an increase or rise
- préstamo --> loan
- conjunto --> collection or group
- escojer --> to choose
- cojo --> weak
- mediana --> median, media --> average

## Supervised learning
Learn a function that maps an input to an output given labelled/known input-output pairs. Two types: Classification and Regression

- Classification (predict class e.g. A, B, C or 0s and 1s)
- Regression (predict quantity e.g. revenue day 7)

Both use input variables e.g. gender, weight to calculate an output variable

### Regression and Classification
Tabular data. Rows are observations, columns are features
- Linear/Logistic Regression (OLS, MLE)
- Regularized linear models (LASSO, Ridge, ElasticNet)
- Random Forest
- Gradient Boosting Machines (XGBoost, LightGBM, Catboost)

Side note:
You can make new variables from a give dataset g.g. change in number of employees from one year to next

Gradient boosting --> Like a random forest where the next tree corrects the errors of previous tree

For the output of the model, it doesn't have to be a binary number. In fact, the model will most likely output a probability. E.g. if probability to default is higher, we can charge them more in order to accept them as a client but mitigate the potential risk

## Unsupervised learning
Find previously unknown patterns in your data without pre-existing labels. Tabular data. Rows are observations, columns are features. Examples include

- Clustering e.g Kmeans
- Dimension reduction (e.g PCA)
- Association (e.g. people who buy X tend to also buy Y)

Side note:
Rather than solve an unsupervised problem, label some data and use supervised learning

Side note:
To run a Kmeans we would
- Choose K
- Standardise feature columns
- Plot observations on axes where each variables is an axis
- Start clusterizing with the first K points being the starting centers
- Add each observation to the axes, find its closest point, add to this cluster and recalculate the center
- In real life? Clusters don't always fall out so easily

## Other types of ML
- Reinforcement learning
- Ensemble methods e.g. bagging and boosting
- Neural nets and deep learning

### Image Recognition
Each observation is a sequence of numbers between 0 and 255 for each colour RGB
- Standard Neural Network (Multi-Layer Perceptron)
- 2D convolutional NN

Side note:
Most people use Keras for image analysis

### Text Sentiment Analysis
Each observation is a text in some language
- Standard Classification with a previous word sparse embedding (NLP: Stop Words, Stemming and Vectorizer)
- 1D convolutional NN
- Recurrent NN (LSTM: Long Short-Term Memory)

Side note:
If you have tabular data --> XGBoost wins Kaggle
If you have Image data --> Deep learning wins Kaggle

## Overfitting and Underfitting

**Overfitting** is when you fit too closely to a particular set of data and fail to generalize to new data or to predict future observations. Normally it contains more parameters than is justified by the data. It is essentially when we have unknowingly fit some of the noise

**Underfitting** is when our model cannot adequately capture the underlying structure of the data. Some parameters that would describe the data are missing. Such a mode would have poor predictive performance

**Bias/Variance tradeoff**
Central problem in supervised learning. Generally speaking, we want a model that accurately captures the regularities in the training data but that also generalizes well to unseen data; High variance models are at risk of overfitting to noisy or unrepresentative training data. In contrast, low variance models are at risk of underfitting, failing to capture important regularities.

### Best practices to avoid overfitting
To Avoid Overfitting: (Hyper)parameter tuning

**Train/Test (or validation) Split**
- Data should be split into 70% training and 30% test
- The data should be stratified i.e. the test data should look a bit like the training data. If we have 60% males and 40% females in the whole dataset, we should see the same split in both training and testing
- Test data  (often called the validation set) is typically "out-of-sample" or "out-of-time". This means that we know the labels but choose to ignore them until the end of the modelling process
- All transformations should be done ONLY using training data and then applied to the test set e.g. when standardising data we do so using the mean and standard deviation of the training set ONLY and apply these same parameters to the test set
- Model tuning i.e. missing data imputation, feature engineering, feature transformations, categorical transformations, should be evaluated and optimized using train-test split or cross validation

To reiterate, whatever you do to them training set, you must to do testing set e.g. imputing missing values, create new features, encoding categorical variables, etc. Test data needs to be the same as training data, i.e same format, columns, everything.


Side notes:
- Fill NAs only with training data. Imagine test data doesn't exist. Fill the test NAs in EXACTLY the same way e.g. if they were filled with the mean of the training set, fill them with the mean of the training setl
- For encoding categorical variables, imagine converting data usage of "High", "Mid", "Low" to number e.g. High = 10, Mid = 7, Low = 3
- It's possible that there are variables/values that appear in your test set that weren't in your train set. You might need to handle those. E.g assign 0 or median or something

**K-fold cross validation**
To properly do hyper-parameter tuning we should use k-fold cross validation. This consists of first holding out our training set until the very end when we have the best hyper-parameters. Then we split our training set into k partitions, and iteratively train a model on each combination of k-1 partitions (as a whole training set) to predict the kth test partition. This will give us a whole column of predictions which we can then compare against the real data to get a score e.g AUC = 80%. Once we have the right hyper-parameters to optimize our metric, in this case AUC, we can retrain the model on the whole training dataset and use it to predict the real holdout test set.

**Hyper-parameter vs. Parameter**
The main difference between hyper-parameters and parameters is that we set the hyper-parameters, e.g. the number of trees in a random forest. The parameters, are what fall out of the model, the parameters that best predict the test data from the training data e.g. at a given node of a decision tree, the best split of revenues is >=$1M or <$1M would be an example of one parameter. We can grid search over hyper-parameters to fine tune the model e.g. 10, 100, 1000 trees

## Stacking vs. Ensemble Models

### Ensemble models
- Bagging
- Boosting

What is an Ensemble model?
- N weak learners (usually trees)
- Each learner is "grown" from different data parts i.e. a subset of the features. Prediction voting or mean (Bagging)
- Each learner corrects the predictions of the predecessor (Boosting)

### Stacking
Create different models and then join the predictions.

In a similar way to k-fold cross-validation, we can apply multiple models to the k-fold split and get a column of predictions for each model. These can then be used as meta-features for a subsequent training/testing round, either with or without the original dataset

For more info, I think [this](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/) is the best explanation and has python code.

We can then take this new "feature" alone or with the old database and run the hyper-parameter tuning all over again.

Side note:
When performing Stacking, pick models that are different e.g. linear, RF, NN


### Questions

Q: What's the difference between train/test hyper parameter tuning and the cross-validation hyper parameter training?
A: Hyper-parameters are what we tune, parameters are what the model spits out as the best parameters for that method to predict the test data with the training data

Q: 2 Options slide 16 --> Don't understand "Go to start with new dataset"
A: It's talking about Stacking i.e. taking the predictions as features in a second-level round of predictions. See main text

Q: How to concatenate metric from k-fold split? average of metric per split?
A: Concatenate, i.e. literally union or join the results into one column. Each train test split will only give you predictions for a k-th of the data. Concatenate them all into one predictions column, compare against the real labels and output a metric e.g AUC

Q: Is it possible to overfit the k-fold? You're testing on 5 validation sets. If you test 20 parameter combinations, and are always checking the same train/test split. Would you change the split each time?
A: Technically yes, it's possible to overfit but it's unlikely. And yes, one way to avoid it would be to change the random seed when generating the train/test split to get different folds


**Notes for next time:**

Regression Metrics
- MAE
- RMSE (square root missing from slide) --> Normally wins. Idea is to minimise
- R-sq or adj-R-sq (adj means taking into account the number of variables used)
- Gini - how good is a model at predicting 0,1 (2*AUC - 1)
- For Interpretability of models we can use SHAP or LIME which we will cover in another lesson.
