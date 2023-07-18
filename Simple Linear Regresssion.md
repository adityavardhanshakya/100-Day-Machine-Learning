# Simple Linear Regression

# Step 1: Data Preprocessing
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 
```

# Step 2: Fitting Simple Linear Regression Model to the training set
 ```python
 from sklearn.linear_model import LinearRegression
 regressor = LinearRegression()
 regressor = regressor.fit(X_train, Y_train)
 ```
 # Step 3: Predecting the Result
 ```python
 Y_pred = regressor.predict(X_test)
 ```
 
 # Step 4: Visualization 
 ## Visualising the Training results
 ```python
 plt.scatter(X_train , Y_train, color = 'red')
 plt.plot(X_train , regressor.predict(X_train), color ='blue')
 ```
 ## Visualizing the test results
 ```python
 plt.scatter(X_test , Y_test, color = 'red')
 plt.plot(X_test , regressor.predict(X_test), color ='blue')
 ```
![Hours vs Score (Test set)](https://github.com/adityavardhanshakya/100-Day-Machine-Learning/assets/75056596/fe8a7efa-e83f-4b26-9c49-aa0df1af28a2)
![Hours vs Score (Training set)](https://github.com/adityavardhanshakya/100-Day-Machine-Learning/assets/75056596/17f00525-87f4-4ca7-8efb-7b618853f770)


 ```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Create a linear regression model and fit it to the training data
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = regressor.predict(X_test)

# Visualize the training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Hours vs Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# Visualize the test set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Hours vs Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

# # Print the coefficients and intercept of the linear regression model
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

 ```
