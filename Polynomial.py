# Importing libraries
# -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# ---------------------
dataset = pd.read_csv('Polynomial-Dataset.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
predicted_vals = []
experience = 6.5  # Experience that needs to be predicted

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
predicted_vals.append(lin_reg_2.predict(poly_reg.fit_transform([[experience]])))

# Fitting Polynomial Regression to the dataset with degree=3
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results with degree=3
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
predicted_vals.append(lin_reg_2.predict(poly_reg.fit_transform([[experience]])))

# Fitting Polynomial Regression to the dataset with degree=4
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results with degree=4
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Polynomial Regression
predicted_vals.append(lin_reg_2.predict(poly_reg.fit_transform([[experience]])))

# Predicting a new result with Linear Regression
predicted_vals.append(lin_reg.predict([[experience]]))

test_data = [experience] * len(predicted_vals)
plt.scatter(test_data, predicted_vals, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Comparison of SLR regression vs Poly Degree results(Degree 2-4)')
plt.xlabel('Experience to be predicted = 6.5')
plt.ylabel('Predicted Values - SLR, Poly Degree - [2, 3, and 4]')
annotations = ['Degree 2', 'Degree 3', 'Degree 4', 'Linear']

for i, label in enumerate(annotations):
    plt.annotate(label, (test_data[i], predicted_vals[i]))
plt.legend((), fontsize=6)
plt.show()

print(predicted_vals)

