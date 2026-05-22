import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b22d1166-efda-45e8-979e-6c3ecfc566fc/houses_poly.csv')
# Assign the variables
X = df[['age']]
y = df['price']
n = 2    # A degree of Polynomial Regression

# Preprocess X
poly = PolynomialFeatures(n)
X_poly = poly.fit_transform(X)

# Build and train the model
model = LinearRegression()
model.fit(X_poly, y)

# Create and preprocess X_new
X_new = np.linspace(0, 125, 200).reshape(-1, 1)
X_new_poly = poly.transform(X_new)

# Predict the target for X_new
y_pred = model.predict(X_new_poly)

# Visualize the result
plt.scatter(X, y, alpha=0.4)
plt.plot(X_new, y_pred, color='orange')
plt.show()

# Print the model's parameters
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)