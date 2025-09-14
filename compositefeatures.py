import pandas as pd

# Example data
df = pd.DataFrame({
    'Height_m': [1.75, 1.80, 1.65],
    'Weight_kg': [70, 80, 60],
    'Price': [10, 15, 20],
    'Quantity': [3, 2, 5]
})

# Create composite features
df['BMI'] = df['Weight_kg'] / (df['Height_m']**2)
df['Revenue'] = df['Price'] * df['Quantity']

print(df)
