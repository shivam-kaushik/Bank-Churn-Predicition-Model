import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained scaler and model
scaler = joblib.load('C:/Users/ishik/Downloads/Ishika Capstone/Ishika Capstone/models/scaler.pkl')
logistic_regression_model = joblib.load('C:/Users/ishik/Downloads/Ishika Capstone/Ishika Capstone/models/logistic_regression_model.pkl')
decision_tree_model = joblib.load('C:/Users/ishik/Downloads/Ishika Capstone/Ishika Capstone/models/decision_tree_model.pkl')
gradient_boosting_model = joblib.load('C:/Users/ishik/Downloads/Ishika Capstone/Ishika Capstone/models/gradient_boosting_model.pkl')
random_forest_model = joblib.load('C:/Users/ishik/Downloads/Ishika Capstone/Ishika Capstone/models/random_forest_model.pkl')

# Expected columns
expected_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male']
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess the data (same steps as training)
        # Encoding categorical variables
        df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
        
        # Add missing columns if necessary (in case the new data doesn't have all possible categories)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Ensure the DataFrame columns are in the correct order
        df = df[expected_columns]
        
        # Scaling numerical features
        df[numerical_features] = scaler.transform(df[numerical_features])
        
        # Create the response
        response = {
            'Logistic Regression Prediction': int(logistic_regression_model.predict(df)[0]),
            'Logistic Regression Probability': float(logistic_regression_model.predict_proba(df)[:, 1]),
            'Decision Tree Prediction': int(decision_tree_model.predict(df)[0]),
            'Decision Tree Probability': float(decision_tree_model.predict_proba(df)[:, 1]),
            'Gradient Boosting Prediction': int(gradient_boosting_model.predict(df)[0]),
            'Gradient Boosting Probability': float(gradient_boosting_model.predict_proba(df)[:, 1]),
            'Random Forest Prediction': int(random_forest_model.predict(df)[0]),
            'Random Forest Probability': float(random_forest_model.predict_proba(df)[:, 1])
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
