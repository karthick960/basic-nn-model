# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![alt text](image.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:KARTHICK K
### Register Number:212222040070
```python
from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials
from google.auth import default
import pandas as pd

!pip install scikit-learn  

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)

auth.authenticate_user()
creds, _ = default()  # Get credentials using the newer google-auth library
gc = gspread.authorize(creds)

worksheet=gc.open('12.8.24 deeplearning').sheet1
data=worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])  # Capitalize 'F' in 'DataFrame'
dataset1 = dataset1.astype({'input': 'float'})  # Change 'Input' to 'input'
dataset1 = dataset1.astype({'output': 'float'})  # Change 'Output' to 'output'

dataset1.head()

x = dataset1[['input']].values  # Change 'Input' to 'input'
y = dataset1[['output']].values  # Change 'Output' to 'output'

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(X_train)

X_train1=Scaler.transform(X_train)

ai_brain=Sequential([
   # Dense(8,input_dim=1,activation='relu'),
    Dense(16,activation='relu'),
    Dense(1)
])

ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=2000)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1=Scaler.transform(X_n1)
ai_brain.predict(X_n1)



```

## Dataset Information

![alt text](image-1.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![alt text](image-2.png)

### Test Data Root Mean Squared Error

![alt text](image-3.png)

### New Sample Data Prediction

![alt text](image-4.png)

## RESULT

To develop a neural network regression model for the given dataset is created sucessfully.
