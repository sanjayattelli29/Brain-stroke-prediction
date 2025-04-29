import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl"
}

models = {}
for name, file in model_files.items():
    if os.path.exists(file):
        with open(file, "rb") as f:
            models[name] = pickle.load(f)

# Load label encoder and scaler
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load metrics
if os.path.exists("model_performance_metrics.csv"):
    metrics_df = pd.read_csv("model_performance_metrics.csv")
else:
    metrics_df = pd.DataFrame()

# Categorical columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

@app.route('/')
def home():
    return jsonify({"message": "Stroke Risk Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input keys
        expected_keys = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                         'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

        if not all(k in data for k in expected_keys):
            return jsonify({"error": "Missing input fields."}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Encode categorical features
        for col in categorical_cols:
            input_df[col] = label_encoder.fit_transform(input_df[col].astype(str))

        # Scale input
        scaled_input = scaler.transform(input_df)

        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(scaled_input)[0]
            result = "Stroke Risk" if prediction == 1 else "No Stroke Risk"
            predictions[model_name] = result

        # Get best model by Accuracy
        best_model = metrics_df.sort_values(by="Accuracy", ascending=False).iloc[0] if not metrics_df.empty else None

        response = {
            "predictions": predictions,
            "final_recommendation": f"Best performing model: {best_model['Model']} with accuracy {best_model['Accuracy']:.2f}" if best_model is not None else "Performance metrics unavailable.",
            "model_metrics": metrics_df.to_dict(orient="records")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
