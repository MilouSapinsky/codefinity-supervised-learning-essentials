import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b22d1166-efda-45e8-979e-6c3ecfc566fc/houseprices.csv')
# Assign the variables
X = df[['age', 'square_feet']]
y = df['price']

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Create X_new
X_new = np.array([[4, 10000], [30, 14000], [70, 16000]])

# Predict instances from X_new and print them
y_pred = model.predict(X_new)
print('Prediction:', np.floor(y_pred)) # np.floor() keeps only the whole part of the numbers

# Print the model parameters
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)