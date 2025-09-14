from sklearn.linear_model import LinearRegression
import numpy as np

# Input features (X)
X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])

# Multiple outputs (Y)
Y = np.array([[2, 3],
              [3, 5],
              [4, 7],
              [5, 9]])

# Create model
model = LinearRegression()
model.fit(X, Y)

# Predict
predictions = model.predict([[5, 6]])
print(predictions)
