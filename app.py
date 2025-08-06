import streamlit as st
import joblib
import os

# Load model and vectorizer
model_path = 'model/scam_model.pkl'
vectorizer_path = 'model/vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("Model or vectorizer file not found. Please run train.py first.")
else:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # App UI
    st.title("🔍 Scam Message Detector")
    st.write("Enter a message to check if it's a **scam** or **safe**.")

    user_input = st.text_area("💬 Enter Message")

    if st.button("Check"):
        if user_input.strip() == "":
            st.warning("Please enter a message to analyze.")
        else:
            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]
            prediction_prob = model.predict_proba(X)[0][1]  # probability of being scam

            st.write(f"📊 Scam Probability: **{prediction_prob:.2f}**")

            if prediction == 1:
                st.error("⚠️ Scam detected!")
            else:
                st.success("✅ Message looks safe.")
