# Lesson 2

### Spanish vocab
- empates --> ties
- fichero --> file
- corchetes --> brackets

## Regression metrics
- MAE - Mean absolute error
- RMSE (square root missing from slide) --> Normally wins. Idea is to minimise this. Most used; computationally simple and default metric for most models
- R-sq or adj-R-sq (adj means taking into account the number of variables used)


## Classification (rank) metrics

### Accuracy ratio

aka **GINI**
aka power index

Calculate probability of class 1  P(1)
Normally category 1 is the bad one
Cumulative accuracy profile (CAP) curve

*Deleted my explanation*

CAP curves explained [here](https://towardsdatascience.com/machine-learning-classifier-evaluation-using-roc-and-cap-curves-7db60fe6b716)

The AR (accuracy ratio) is the area between my model and random model as a proportion of the perfect model

### AUC

-- Fill later

### Kolmogorov - Smirnov (Two Sample)

Can be used to test if a distribution matches the normal curve, but also can be used to compare our curves from 0s and 1s
Perfect would have maximum distance of 1

K-S <= AR <= AUC

### Somers' D (Kendall's Tau ignoring ties)
Check in pairs if one prof gave a higher score to B than A and compare to the other. If Prof 1: B > A and Prof 2: B > A, +1. If Prof 1: B > A and Prof 2: A > B, -1.

AR == Sommer's D when Prof 2 is 0/1 == Kendall's Tau without ties

*metrics.py*

## Logistic Regression
Not for regression. For Classification

logit == log odds
P(event)/1-P(event)

Let Pr(y = 1 | X) = p(X)

Sigmoid function

model.coef_ --> not really coefficients. need to take exponential, no?

weight of evidence

*logistic.py*

## Linear Models
- Linear Regression

Minimise RMSE/Least Squares Method

*linear.py* (continued in lesson 3)
