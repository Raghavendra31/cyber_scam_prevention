from flask import Flask, request, jsonify
import joblib, re
from cryptography.fernet import Fernet

app = Flask(__name__)

# ===== Load Model & Vectorizer =====
try:
    model = joblib.load("model/scam_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    exit(1)

# ===== Load Encryption Key =====
with open("secret.key", "rb") as f:
    secret_key = f.read()
fernet = Fernet(secret_key)

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
            return max(probability, 0.6)
    return probability

@app.route("/check", methods=["POST"])
def check_scam():
    try:
        data = request.get_json(force=True)
        encrypted_message = data.get("data", "")

        if not encrypted_message:
            return jsonify({"error": "No encrypted data provided"}), 400

        # 📥 Show encrypted incoming message
        print("📥 Encrypted incoming message:", encrypted_message)

        # 🔑 Decrypt message
        decrypted_message = fernet.decrypt(encrypted_message.encode()).decode()

        # Clean text
        cleaned_message = clean_text(decrypted_message)
        X = vectorizer.transform([cleaned_message])

        # Predict probability
        probability = model.predict_proba(X)[0][1]

        # Apply keyword boost
        probability = keyword_boost(cleaned_message, probability)

        threshold = 0.6
        prediction = 1 if probability >= threshold else 0

        # Logs
        print("\n🔍 Decrypted message:", decrypted_message)
        print("🧹 Cleaned:", cleaned_message)
        print(f"🧠 Prediction: {prediction} | Scam Probability: {probability:.4f}")

        response = {
            "original_message": decrypted_message,
            "cleaned_message": cleaned_message,
            "prediction": prediction,
            "scam_probability": round(probability, 4)
        }

        # 📤 Show outgoing response JSON
        print("📤 Response JSON:", response)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

#https://2669f78ea775.ngrok-free.app