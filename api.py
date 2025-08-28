from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# ===== Load Model & Vectorizer =====
try:
    model = joblib.load("model/scam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    exit(1)

# ===== Text Cleaning Function =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # keep alphanumeric & spaces
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# ===== Scam Keyword List =====
SCAM_KEYWORDS = [
    "invest", "profit", "earn", "deposit", "trading", 
    "funds", "cash", "lottery", "win", "reward", "urgent", 
    "click here", "offer", "account"
]

def keyword_boost(text, probability):
    """Boost probability if scam keywords are found"""
    for word in SCAM_KEYWORDS:
        if word in text:
            # Ensure probability is at least 0.6
            return max(probability, 0.6)
    return probability

@app.route("/check", methods=["POST"])
def check_scam():
    data = request.get_json(force=True)
    message = data.get("message", "")

    # Clean text
    cleaned_message = clean_text(message)
    X = vectorizer.transform([cleaned_message])

    # Predict probability
    probability = model.predict_proba(X)[0][1]

    # Apply keyword boost
    probability = keyword_boost(cleaned_message, probability)

    # ===== Updated threshold logic =====
    threshold = 0.6  # probability >= 0.5 → scam
    prediction = 1 if probability >= threshold else 0

    # Logs
    print("\n🔍 Received:", message)
    print("🧹 Cleaned:", cleaned_message)
    print(f"🧠 Prediction: {prediction} | Scam Probability: {probability:.4f}")

    return jsonify({
        "original_message": message,
        "cleaned_message": cleaned_message,
        "prediction": prediction,
        "scam_probability": round(probability, 4)
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)