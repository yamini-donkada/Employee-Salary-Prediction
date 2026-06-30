import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("salary_data.csv")

# Input and output
X = data[["Experience"]]
y = data["Salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict salary for 8 years experience
years = float(input("Enter years of experience: "))

experience = pd.DataFrame([[years]], columns=["Experience"])
predicted_salary = model.predict(experience)
print("Predicted Salary:", predicted_salary[0])
