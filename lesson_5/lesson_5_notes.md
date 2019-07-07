# Lesson 5

### Spanish vocab
- moroso
- competencia
- particulares
- ajustar
- mora
- forma de sierra
- anclar  
- comerse con patatas ?
- capa
- nabla --> delta?

## Calibration

Always calibrate with train

Can change central tendency if we think the economy is bad for example we can increase all PD by 5% o algo asi

Can re-calibrate when we have too many 0s by throwing away 0s at random. But we need to recalibrate again because prob of default will be higher that in real life and you'll want to properly predict probability on train with real PD.

## Confusion Matrix
We can set the cut off to minimise what we're interested in, FP or FN.

We can check the loss or gain that we would get in each situation e.g. money gained from good companies - money lost from bad companies. Also don't forget to put opportunity cost

Calculate the maximum gain from the correct cut of probabilities in the confusion matrix

We can heavily penalise incorrect classifications of one class if we want.

Companies sometimes operate with a loss at the beginning, get more clients, and then increase the price for everyone.

## Classic Approach vs. Tasche

Tasche is state of the art

## Validation

PSI Population stability index

## Black box models

SHAP
LIME

## NN

Inputs e.g. features. Transpose of rows

In some cases, Stacking works worse than Catboost

LTSM -> for sentiment analysis Recurrent NN

Needs numerical inputs. Need to transform them

Bias is like intercept of regression

Sigmoid is the logistic function. A NN without any hidden layers (only input and output) and a sigmoid activation function IS a logistic regression

Initialise weights with random number

Activation functions are hyper parameters. Should be defined before training.

Gradient descent is how we train the NN

Learning rate (eta) is how much you move when doing gradient descent. Too small and we can hit a local min

TF - Google
Pytorch - Facebook
Keras - library on top of TF to make it easier to train

batch size --> how many samples to train each time

epochs --> How many times to loop over whole dataset when i've done all the batches in the dataset

learning rate --> size of pasos

## Convolutional NN

Softmax of probabilities of being 0-9

2D CNN - take 3x3 matrix ("kernel") e.g. 101,010,101, pass over image one cell at a time multiply by pixel value and add?

Max pooling 2D

Can rotate image and see which is better. Or we can train on rotated images but would make it complicated

## Homework
Quasi moment matching
OSRF one single risk factor
Softmax?
Keras LSTM tutorial - recursive, backwards connections in NN -> Text analysis

## Questions
So a NN needs a target/labels?
How do you tell it that it needs to predict 0-9?

- 1 week to check things and ask Jordi

- How did you execute the scraper?
cd Sentiment Analysis
cd expansionnew
scrapy crawl expansionnew

- Send Jordi code and submission name in Kaggle
