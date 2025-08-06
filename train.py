import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
data_path = os.path.join("data", "scams.csv")
df = pd.read_csv(data_path)

# Standardize and clean data
df.columns = df.columns.str.lower()
df['label'] = df['label'].str.strip().str.lower()
df = df[df['label'].isin(['safe', 'scam'])]  # Filter only valid labels
df.dropna(subset=['message', 'label'], inplace=True)

# Encode labels: scam = 1, safe = 0
df['label'] = df['label'].map({'safe': 0, 'scam': 1})

# Show label distribution
print("🔍 Label distribution:\n", df['label'].value_counts())

# Train/test split
X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/scam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Model and vectorizer saved to /model/")
