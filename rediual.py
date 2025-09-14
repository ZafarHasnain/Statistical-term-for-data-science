import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.5, 3.8, 3.0, 4.5, 5.1])

# Model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Residuals
residuals = y - y_pred

# Plot
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
