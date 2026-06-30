from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv(r"C:\Users\ADMIN\Desktop\Employee_Salary_prediction\dataset\salary_data.csv")

# Train model
X = data[["Experience"]]
y = data["Salary"]

model = LinearRegression()
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    experience = int(request.form["experience"])

    prediction = model.predict([[experience]])

    return render_template(
        "result.html",
        experience=experience,
        salary=round(prediction[0], 2)
    )

if __name__ == "__main__":
    app.run(debug=True)