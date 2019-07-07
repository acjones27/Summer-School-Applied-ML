# Lesson 4

### Spanish vocab
- por donde van los tiros

## Non-Linear Models
### CART (Classification and Regression Trees)

Logloss or Gini impurity to find the optimal cut for Classification Trees
DT hyperparams --> tree depth, number of leaves, stop when leaf has fewer than X observations etc
Can minimise the RMSE at each point for Regression Trees

Python can print out trees with values

### Random Forest
They don't use 1 tree ever. RF is a combination of trees but they have to be different trees, otherwise the same trees will always fall out. Since the DT gives the optimal split.

Can let the trees see a different % of columns, % of observations.
Make the DTs weak. They themselves can't learn a lot, but together yes.

Find the optimal params with train-test/cross-val.


### Gradient Boosting

If the column is T/F, the split is T/F. First prediction is average of group e.g all Ts or all Fs.

Take this prediction as input to next split

### CatBoost
If you don't have categorical vars, maybe LightGBM

Params:
Learning rate --> 0.1, 0.001, 0.002
Gradient Boosting --> Tries to learn from previous trees

## Notes

Option A: one train test split of 30%
Option B: k-fold cross validation
Option C: first 1-fold then k-fold, for expensive models

Check part of makepred==True. It's the stacking part, the part of second level models where we take the predictions of the models as inputs to the next one

Tasche is better than Binomial for Calibration.

https://www.kaggle.com/nholloway/catboost-v-xgboost-v-lightgbm

### Questions
- We upload the probabilities? YES
- For the DT, is the best split determined by reducing the RMSE of the diff between the real value and the average?
- How bad is it to have correlated features? Will it affect the prediction?

### Homework
- Video of gradient boosting (presentation)
- Class imbalance? What to do?
