from flask import Flask, render_react, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and artifacts
MODEL_PATH = os.path.join('model', 'titanic_survival_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
ENCODER_PATH = os.path.join('model', 'label_encoder.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        pclass = int(request.form['pclass'])
        sex = request.form['sex'] # 'male' or 'female'
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        fare = float(request.form['fare'])

        # Preprocess input
        sex_encoded = label_encoder.transform([sex])[0]
        
        # Create DataFrame for scaling and prediction
        input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, fare]], 
                                 columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'])
        
        # Scale numerical features
        input_data[['Age', 'Fare']] = scaler.transform(input_data[['Age', 'Fare']])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        result = "Survived" if prediction == 1 else "Did Not Survive"
        prob_percent = round(probability * 100, 2)

        return render_template('index.html', 
                               prediction_text=f'Result: {result}',
                               probability_text=f'Probability of Survival: {prob_percent}%',
                               result_class='survived' if prediction == 1 else 'not-survived')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
