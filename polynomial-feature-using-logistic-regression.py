import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# Dataset
X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])

# Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_poly, y)

# Prediction
value = float(input("Enter value: "))
value_poly = poly.transform([[value]])

prediction = model.predict(value_poly)

print("Prediction:", prediction[0])