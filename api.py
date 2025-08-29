from flask import Flask, request, jsonify
import joblib, re
from cryptography.fernet import Fernet
import os # Import os for path checking

app = Flask(__name__)

# ===== Load Model & Vectorizer =====
try:
    # Ensure the model directory exists or adjust paths if necessary
    if not os.path.exists('model/scam_model.pkl') or not os.path.exists('model/vectorizer.pkl'):
        # Correct path for scam_model.pkl based on train.py output
        if not os.path.exists('scam_model.pkl'): # Check in root first
             raise FileNotFoundError("scam_model.pkl not found. Make sure 'train.py' was run and outputted it correctly.")
        if not os.path.exists('model/vectorizer.pkl'): # Check in model/ for vectorizer
             raise FileNotFoundError("model/vectorizer.pkl not found. Make sure 'train.py' was run and outputted it correctly.")

    model = joblib.load("scam_model.pkl") # Assuming scam_model.pkl is in the root as per train.py
    vectorizer = joblib.load("model/vectorizer.pkl")
    print("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    # It's better to exit if core components can't load
    exit(1)

# ===== Load Encryption Key =====
try:
    with open("secret.key", "rb") as f:
        secret_key = f.read()
    fernet = Fernet(secret_key)
    print("✅ Encryption key loaded successfully.")
except FileNotFoundError:
    print("❌ Error: secret.key not found. Please generate it using 'from cryptography.fernet import Fernet; key = Fernet.generate_key(); with open(\"secret.key\", \"wb\") as key_file: key_file.write(key)'")
    exit(1)
except Exception as e:
    print(f"❌ Error loading encryption key: {e}")
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
    "click here", "offer", "account", "verify", "link", "deactivated",
    "blocked", "penalty", "claim", "prize", "congratulations",
    "exclusive", "free", "limited", "payout", "bonus"
]

def keyword_boost(text, probability):
    """Boost probability if scam keywords are found"""
    initial_probability = probability
    for word in SCAM_KEYWORDS:
        if word in text:
            # Increase the probability if a keyword is found, but cap it.
            # A simple boost could be:
            probability = max(probability, 0.6) # Ensure a baseline scam probability if a keyword is present
            # For a more nuanced boost, you could add a small amount, e.g., probability += 0.1
    
    # If a scam keyword is present and the initial probability was low, boost it
    # This prevents benign messages with common words (like "account") from being flagged too easily
    # while still boosting clearly suspicious messages.
    if probability > initial_probability and initial_probability < 0.5: # Only boost if keyword made a difference and it was originally uncertain
        return max(probability, 0.7) # Give a stronger boost if it seems like a potential scam
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
        print(f"❌ Error in /check endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # In a production environment, debug=False is recommended for security and performance
    # Also, consider using a production-ready WSGI server like Gunicorn or uWSGI
    app.run(debug=True, host="0.0.0.0", port=5000)