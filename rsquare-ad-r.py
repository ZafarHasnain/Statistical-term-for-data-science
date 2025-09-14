import numpy as np
import statsmodels.api as sm

# Sample data
X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5],
              [5, 6]])
y = np.array([2.3, 3.5, 3.0, 4.8, 5.6])

# Add constant term (intercept)
X = sm.add_constant(X)

# Build model
model = sm.OLS(y, X).fit()

# R-squared and Adjusted R-squared
print("R-squared:", model.rsquared)
print("Adjusted R-squared:", model.rsquared_adj)
