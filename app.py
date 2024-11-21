from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize FastAPI
app = FastAPI()

# Define the input data structure (used for validation in POST)
class DiabetesInput:
    def __init__(self, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

# Root route for the form (GET request)
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diabetes Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                padding: 20px;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            form {
                max-width: 500px;
                margin: 0 auto;
                padding: 20px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            label {
                display: block;
                margin: 10px 0 5px;
            }
            input {
                width: 100%;
                padding: 8px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                width: 100%;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                max-width: 500px;
                margin: 20px auto;
                padding: 20px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            ul li {
                margin: 5px 0;
            }
        </style>
    </head>
    <body>
        <h1>Diabetes Prediction Form</h1>
        <form action="/" method="post">
            <label for="Pregnancies">Pregnancies:</label>
            <input type="number" name="Pregnancies" required><br>
            <label for="Glucose">Glucose:</label>
            <input type="number" name="Glucose" required><br>
            <label for="BloodPressure">Blood Pressure:</label>
            <input type="number" name="BloodPressure" required><br>
            <label for="SkinThickness">Skin Thickness:</label>
            <input type="number" name="SkinThickness" required><br>
            <label for="Insulin">Insulin:</label>
            <input type="number" name="Insulin" required><br>
            <label for="BMI">BMI:</label>
            <input type="number" step="0.1" name="BMI" required><br>
            <label for="DiabetesPedigreeFunction">Diabetes Pedigree Function:</label>
            <input type="number" step="0.1" name="DiabetesPedigreeFunction" required><br>
            <label for="Age">Age:</label>
            <input type="number" name="Age" required><br><br>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """

# Handle form submission (POST request)
# Handle form submission (POST request)
@app.post("/", response_class=HTMLResponse)
def predict_form(
    Pregnancies: int = Form(...),
    Glucose: int = Form(...),
    BloodPressure: int = Form(...),
    SkinThickness: int = Form(...),
    Insulin: int = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...),
):
    # Convert input data to numpy array
    data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Scale the input data using the saved scaler
    data_scaled = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data_scaled)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

    # Return a response with the result and input details
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                padding: 20px;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .result {{
                max-width: 500px;
                margin: 20px auto;
                padding: 20px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            ul {{
                list-style-type: none;
                padding: 0;
            }}
            ul li {{
                margin: 5px 0;
            }}
            a {{
                display: block;
                text-align: center;
                margin-top: 20px;
                text-decoration: none;
                color: #4CAF50;
                font-weight: bold;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="result">
            <h1>Prediction Result: {result}</h1>
            <p>Details:</p>
            <ul>
                <li>Pregnancies: {Pregnancies}</li>
                <li>Glucose: {Glucose}</li>
                <li>BloodPressure: {BloodPressure}</li>
                <li>SkinThickness: {SkinThickness}</li>
                <li>Insulin: {Insulin}</li>
                <li>BMI: {BMI}</li>
                <li>Diabetes Pedigree Function: {DiabetesPedigreeFunction}</li>
                <li>Age: {Age}</li>
            </ul>
            <a href="/">Back to Form</a>
        </div>
    </body>
    </html>
    """

