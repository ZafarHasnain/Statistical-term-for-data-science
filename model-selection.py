from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=5, noise=10)

# Linear Regression
lr = LinearRegression()
lr_score = cross_val_score(lr, X, y, cv=5, scoring='r2').mean()

# Random Forest
rf = RandomForestRegressor()
rf_score = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()

print(f"Linear Regression R²: {lr_score:.2f}")
print(f"Random Forest R²: {rf_score:.2f}")
