import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset (House size in square feet vs price)
size = np.array([500, 800, 1000, 1200, 1500, 1800]).reshape(-1,1)
price = np.array([100000, 150000, 200000, 230000, 300000, 350000])

# Model
model = LinearRegression()

# Train model
model.fit(size, price)

# Prediction
house_size = float(input("Enter house size (sq ft): "))
predicted_price = model.predict([[house_size]])

print("Predicted House Price:", predicted_price[0])