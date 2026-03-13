import numpy as np
from sklearn.linear_model import LinearRegression

hours = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
marks = np.array([10,20,30,40,50,60,70,80,90,95])

model = LinearRegression()
model.fit(hours, marks)

study_hours = float(input("Enter study hours: "))
predicted_marks = model.predict([[study_hours]])

print("Predicted Marks:", predicted_marks[0])