from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# Paths
MODEL_PATH = 'scam_model.pkl'
VECTORIZER_PATH = 'model/vectorizer.pkl'

# Load model & vectorizer
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("❌ Model/vectorizer not found. Run train.py first.")

print("✅ Loading model and vectorizer...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("✅ Model and vectorizer loaded successfully.")

def preprocess_text(text):
    """Apply same preprocessing as in train.py."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

@app.route('/check', methods=['POST'])
def check_scam():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")

        if not message.strip():
            return jsonify({"error": "No message provided"}), 400

        # Preprocess
        processed_msg = preprocess_text(message)

        # Transform & Predict
        X = vectorizer.transform([processed_msg])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        print(f"\n🔍 Received: {message}")
        print(f"🔧 Preprocessed: {processed_msg}")
        print(f"🧠 Prediction: {prediction} | Scam Probability: {probability:.2f}")

        return jsonify({
            "original_message": message,
            "processed_message": processed_msg,
            "prediction": int(prediction),
            "probability": round(float(probability), 2)
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
