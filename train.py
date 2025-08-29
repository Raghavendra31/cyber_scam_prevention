import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import re
import os

# --- 1. Load Data ---
print("✅ Loading data from scams.csv...")
try:
    df = pd.read_csv('data/scamp15k_augmented.csv')
    if df.empty:
        raise ValueError("scams.csv is empty. Please add data.")
except FileNotFoundError:
    print("❌ Error: scams.csv not found. Make sure the file is in the same directory.")
    exit()
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# --- 2. Preprocessing Function ---
def preprocess_text(text):
    """Clean text to remove punctuation and extra spaces."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

df['message'] = df['message'].apply(preprocess_text)
print("✅ Data preprocessing complete.")

# --- The Fix ---
# Convert all values in the 'label' column to a string type.
df['label'] = df['label'].astype(str)
# --- End of Fix ---

# --- 3. Split Data ---
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Vectorize Text (TF-IDF) ---
# Transform text into numerical vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("✅ Text vectorization complete.")

# --- 5. Train the Model (Multinomial Naive Bayes) ---
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("✅ Model training complete.")

# --- 6. Evaluate Model Performance ---
y_pred = model.predict(X_test_vec)
print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))

# --- 7. Save the Model and Vectorizer ---
# Create 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model, 'scam_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
print("\n✅ Model and vectorizer saved as 'scam_model.pkl' and 'model/vectorizer.pkl'.")