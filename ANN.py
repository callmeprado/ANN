#importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import seaborn as sns

#importing Data set
dataset = pd.read_csv('Churn_Modelling.csv')
corr = dataset.corr()
sns.heatmap(corr) 

X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:,13].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x1 = LabelEncoder()
X[:, 1] = labelencoder_x1.fit_transform(X[:, 1])    #for the countries 

labelencoder_x2 = LabelEncoder()
X[:, 2] = labelencoder_x2.fit_transform(X[:, 2])    #for the genders 

onehotencoder = OneHotEncoder(categorical_features = [1])   #making dummy varialbes for index 1 
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]                                        #removing the first column to avoid dummy variable trap 


#Splitting into Train & Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state = 0 )

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)

# --------------------------- MAKING THE ARTIFICIAL NEURAL NETWORK ------------------------------

#Importing Keras libraries and packages 
import keras 
from keras.models import Sequential 
from keras.layers import Dense  ''' ANN step1 : to randomly initialize weights to a no close to 0,
                                    done by Dense()'''

#Initialize the ANN 
classifier = Sequential() #this object is nothing but the future ANN that is going to classify our obs.

# Adding input layer and first hidden layer 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11  ))
 
# Adding 2nd input layer 
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding output layer 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 

'''If you're dealing with a dep variable that has more than 2 categories, you'll have to change the *units*
to the number of desired outputs ...... and *activation* to softmax'''

#Compiling the ANN 
'''Optimizer = the algorithm you wanna use to find the  optimal set of weights in the NN'''
'''loss = loss function within the stochastic gradient descent '''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting ANN to training set 
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

# ------------------------------ MAKING THE PREDICTIONS ---------------------------------

#Predicting Test set results 
y_pred = classifier.predict(x_test)  

#Convertiring y_pred probabilities in to 
y_pred = (y_pred > 0.5)

#Calling confusion matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

type(y_test)