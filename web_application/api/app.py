import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import boto3
import joblib


# Flask app initialization
app = Flask(__name__)
CORS(app)

# AWS S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET")
MODEL_PKL = os.environ.get("MODEL_PKL")


def load_model_from_s3(bucket_name, model_key):
    """
    Downloads the model file from S3 and loads it using joblib.
    """
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile() as tmp_file:
        s3.download_fileobj(bucket_name, model_key, tmp_file)
        tmp_file.seek(0)
        model = joblib.load(tmp_file)
    return model


@app.route("/predict", methods=["OPTIONS", "POST"])
def predict():
    """
    Collect JSON data, transform it into a feature vector, and make a prediction.
    """
    if request.method == "OPTIONS":
        # Preflight response for CORS
        response = app.response_class()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    # Parse JSON data from the request
    data = request.json
    income = data.get("income")
    num_children = data.get("children", 0)
    own_car = 1 if data.get("ownCar", False) else 0
    own_house = 1 if data.get("ownHouse", False) else 0

    # Log the received data for debugging
    print("DEBUG: Form Data")
    print(f"    income:       {income}")
    print(f"    num_children: {num_children}")
    print(f"    own_car:      {own_car}")
    print(f"    own_house:    {own_house}")

    # Convert values to numeric (floats or ints) as needed
    features = np.array([
        int(num_children),
        int(income),
        int(own_car),
        int(own_house)
    ]).reshape(1, -1)  # shape becomes (1, 5)

    # Use your model to predict
    prediction = int(model.predict(features)[0])
    result_str = "Approved" if prediction == 1 else "Rejected"

    print("DEBUG: Prediction Result")
    print(f"    prediction: {prediction}")
    print(f"    result_str: {result_str}")

    return jsonify({
        "prediction": prediction,
        "result_str": result_str,
        "message": f"Model Output (Target): {prediction} ({result_str})"
    })


if __name__ == "__main__":
    print(f"DEBUG: Loading model from: {S3_BUCKET}")
    model = load_model_from_s3(S3_BUCKET, MODEL_PKL)
    print("DEBUG: Model loaded successfully!")
    app.run(host="0.0.0.0", port=5000, debug=True)
