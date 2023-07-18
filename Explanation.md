## Simple Linear Regression - README

This code demonstrates a simple linear regression model using Python. Here's a breakdown of the code's functionality:

1. **Importing Necessary Libraries:**
   - `pandas` (imported as `pd`) is used for data manipulation and analysis.
   - `numpy` (imported as `np`) is used for numerical operations.
   - `matplotlib.pyplot` (imported as `plt`) is used for data visualization.
   - `train_test_split` from `sklearn.model_selection` is used to split the dataset into training and testing sets.
   - `LinearRegression` from `sklearn.linear_model` is used to create a linear regression model.

2. **Loading the Dataset:**
   - The code loads the dataset from the CSV file `'studentscores.csv'` using `pd.read_csv`.
   - The values of the first column are assigned to the variable `X`, and the values of the second column are assigned to `Y`.

3. **Splitting the Dataset:**
   - The code uses `train_test_split` to split `X` and `Y` into training and testing sets.
   - The test size is set to 0.25, indicating that 25% of the data will be used for testing.

4. **Creating and Fitting the Linear Regression Model:**
   - An instance of `LinearRegression` is created and assigned to the variable `regressor`.
   - The model is fitted to the training data using `regressor.fit(X_train, Y_train)`.

5. **Making Predictions on the Test Set:**
   - The code uses `regressor.predict(X_test)` to make predictions on the test data.
   - The predicted values are stored in the variable `Y_pred`.

6. **Visualizing the Training Set Results:**
   - The code creates a scatter plot of the training data points using `plt.scatter(X_train, Y_train, color='red')`.
   - It plots the regression line using `plt.plot(X_train, regressor.predict(X_train), color='blue')`.
   - The title, x-axis label, and y-axis label are set using `plt.title`, `plt.xlabel`, and `plt.ylabel`, respectively.
   - The plot is displayed using `plt.show()`.

7. **Visualizing the Test Set Results:**
   - The code creates a scatter plot of the test data points using `plt.scatter(X_test, Y_test, color='red')`.
   - It plots the regression line (using the same line as in the training set plot) using `plt.plot(X_train, regressor.predict(X_train), color='blue')`.
   - The title, x-axis label, and y-axis label are set using `plt.title`, `plt.xlabel`, and `plt.ylabel`, respectively.
   - The plot is displayed using `plt.show()`.

8. **Printing the Coefficients and Intercept:**
   - The code prints the coefficients (slope) of the linear regression model using `regressor.coef_`.
   - It also prints the intercept (y-intercept) of the linear regression model using `regressor.intercept_`.

This code performs a simple linear regression on the `'studentscores.csv'` dataset. It splits the data into training and testing sets, fits a linear regression model to the training data, makes predictions on the test set, and visualizes the results.
