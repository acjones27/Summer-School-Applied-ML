# Lesson 3

### Spanish vocab
- prestamo --> loan
- azar --> chance
- bucle --> loop

## Feature Engineering
- **Missing imputation**. Should always do it.
- Univariant analysis for feature selection
    - Column by column, check correlation, graphs, r^2 etc
    - If variables are really correlated, you should only end up with 1
    - VARCLUS in SAS, cluster features that are correlated but between-clusters not correlated, and select 1 representative feature from each cluster
- Segmentation
    - Statistical or political reasons
        - If you have a variable that's really predictive, better to segment the users on that column and do 2 models, one for each group e.g. dias sin pagar, so that column doesn't just take over the whole model
        - Find them by doing Univariant analysis, if one has a Gini of 30-40% alone, better to split on that variable.
    - Split over time e.g. one model per year and then use bagging to combine the results
- Linear transformation
    - **WoE for logistic**
    - Mean for OLS
- Categorical variables
    - To Numeric e.g. median/average of another column
    - To Dummies (not recommended) e.g. one column per each value with 1/0s. Don't forget to only make N-1 if there are N values as the Nth is a linear combination of the rest.
- **New features** (Alerts & more)
    - **EWS** (early warning systems). For binary classifications
        - Create new features with 1/0s, using decision tree to cut variable in the best place
        - Gini impurity / Gini gain --> How to best split two groups
        - Label goods (0) and bads (1)
        - Filter goods, check real value and check mean i.e. % of bads
        - Filter bads, check real value and check mean i.e. % of bads
        - Compare to total % of bads
        - Calculate % of bads from bad side after cut * 100 / % of bads in total train. e.g. Real % of bads is 5%, % of bads after split is 15%, the value is 300% (TMR?). It's 3 x more likely to fail
        - Then we can put all the alerts together and count how many alerts a user has
        - Can group them based on magnitude increase e.g. all companies with > 5x probability of default go into graves, between 2-5x M, etc
        - Apply same optimal cut from train to test. IMPORTANT.
        - Can split more to create different alerts e.g. i30, i60, i90, +90 --> segmentation
- Outliers

## Missing imputation

Can put zeros, average of column.
Better way is implemented in the code *linear.py*
- The blue line in the graph is the % of 1s in each bucket. If the missing bucket has a lot of 1s, we would put the value that's similar to other buckets with lots of 1s.
- Or can put a number more than max or less than min (good for trees)

Have to impute in the same way in test. IMPORTANT. to assure ourselves that we're picking a min that's less than the min in test, we can pick a value much less than min in train e.g. 100 x less than min. min = 100, new min = 0.0001. or max = 10000, new max = 5000000

For the a column that looks more like a value between min/max, you could put the median. This will work well for regression. But if you're using trees, min/max works better - Jordi

These are useful for columns with min number of missings e.g. 30 in code

For columns with few missings, can do KNN. Find KNN based on other columns, fill with average of NNs value for missing columns

**fancyimpute** python package nice ways to impute missings

## WoE (weight of evidence)
Take same buckets as last time i.e. even number of observations in each one is even
Number of events (1) and non-events (0s)
WoE is like a combination of % of events over total events, and % of non-events over total non-events
Only for logistic regression

It's possible to overfit here. Need to check the variables and throw away the noisy ones. Does it make sense? if not, throw it. if so, WoE will work.

## Homework
Categorical -> Impute with new category e.g. "Missing"
Check WoE code
Review K-fold, LASSO, RIDGE



### Side notes:
Catboost is his favourite
Can create distributions of each group e.g.
- one dist for users with 1 (x-axis is the variable value, y-axis is count of users with that x-value)
- can pick the best value to separate the two distributions (this is essentially AUC)
- Can find the probability of a 1 appearing to left/right of threshold by counting number of 1s on each side divided by total
- Decision tree with depth 1 of one variable will tell us the best split.
- NPS - marketing


## Questions

- TMR? Relative default rate Taza de ... algo
TMR/Act is a good metric, Act is number of people that have a 1 in that category
- What do you recommend for outliers? How to detect? How to deal? E.g. in my company, the outliers are the ones we care about, the big payers. Can't exclude them but they change the result
- If i put min-1 or max+1 for NaNs, what if they look like some value in the middle? Should i put something in the middle? slide 26

Class imbalance?
-
