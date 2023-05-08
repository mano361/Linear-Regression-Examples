# Importing libraries
# -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# ---------------------
dataset = pd.read_csv('Multiple-Linear-Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# -------------------------
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

location_encoder = ColumnTransformer(transformers=[('Location', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(location_encoder.fit_transform(X))

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
# --------------------------------------------------------
from sklearn.model_selection import train_test_split
# For Product_1 and Product_3 - X[:, [2, 4]]
# For Product_3 and Location(One Hot Encoded) - X[:, [0, 1, 4]]
X_train, X_test, y_train, y_test = train_test_split(X[:, [2, 4]], y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

multi_linear_regression = LinearRegression()
multi_linear_regression.fit(X_train, y_train)

# Predicting the Test set results
y_pred = multi_linear_regression.predict(X_test)

# For calculating the R2 Score
from sklearn.metrics import r2_score
print("R2 Score of the built Model", r2_score(y_test, y_pred))

# Building the optimal model using Backward Elimination
import statsmodels.api as sm

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

print(X[:, [0, 3]])

