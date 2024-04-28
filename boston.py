import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from sklearn import preprocessing

(train_x,train_y),(test_x,test_y) = boston_housing.load_data()

print("train shape :",train_x.shape)
print("Test shape :",test_x.shape)
print("Actual Train Output",train_y.shape)
print("Actual Test Output",test_y.shape)


train_x

train_x[0]

test_x[0]

train_y[0]

train_x=preprocessing.normalize(train_x)
test_x=preprocessing.normalize(test_x)

train_x[0]

train_y[0]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
def HousePricePredictionModel():
  model=Sequential()
  model.add(Dense(128,activation='relu',input_shape=(train_x[0].shape)))
  model.add(Dense(64,activation='relu'))
  model.add(Dense(32,activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
  return model


import numpy as np
k=4 #sets the number of folds (k) to 4
num_val_samples= len(train_x) #determines the number of validation samples
num_epochs = 100 #sets the number of epochs (complete passes through the entire training dataset)
all_scores = [] #initializes an empty list to store validation scores

model = HousePricePredictionModel() #create an instance of the model and train it on the training data for a specified number of epochs
history=model.fit(x=train_x,y=train_y,epochs= num_epochs,batch_size=1,verbose=1,validation_data=(test_x,test_y)) #fit() method is used to “learn” from the data. It finds a function that best fits the provided data

## We will need to convert train and test data using pandas

test = [[0.02675675, 0.        , 0.02677953, 0.        , 0.0010046 ,
        0.00951931, 0.14795322, 0.0027145 , 0.03550877, 0.98536841,
        0.02988655, 0.04031725, 0.04298041]]
print("Actual Output: ", train_y[0])
print("Predicted Output: ", model.predict(test)) # use the trained model to predict the house price
