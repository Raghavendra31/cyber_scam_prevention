# filename: api.py
from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model_path = 'model/scam_model.pkl'
vectorizer_path = 'model/vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise Exception("Model or vectorizer file not found. Please run train.py first.")

print("✅ Loading model and vectorizer...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
print("✅ Model and vectorizer loaded successfully.")

@app.route('/check', methods=['POST'])
def check_scam():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Transform and predict
    X = vectorizer.transform([message])
    prediction = model.predict(X)[0]
    prediction_prob = model.predict_proba(X)[0][1]

    # Debug print for terminal
    print("🔍 Message received:", message)
    print("🧠 Prediction:", prediction, "| Scam Probability:", prediction_prob)

    return jsonify({
        "prediction": int(prediction),
        "probability": round(float(prediction_prob), 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
