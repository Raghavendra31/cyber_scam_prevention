from flask import Flask, request, jsonify
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from Flutter

# Paths for model and vectorizer
model_path = os.path.join("model/scam_model.pkl")
vectorizer_path = os.path.join("model/vectorizer.pkl")

# Load model and vectorizer
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/check', methods=['POST'])
def check_scam():
    try:
        data = request.get_json()
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Transform and predict
        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]

        # Map prediction
        label = 'scam' if prediction == 1 else 'safe'
        return jsonify({'result': label})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "✅ Scam Detector API Running"

if __name__ == '__main__':
    app.run(debug=True, port=5000)


# ngrok http 5000
