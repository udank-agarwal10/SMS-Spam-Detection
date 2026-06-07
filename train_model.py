"""
Retrain SMS Spam Detection Model
- Combines both datasets (spam.csv + spam_ham_india.csv)
- Uses improved TF-IDF features (bigrams, more features)
- Uses Logistic Regression with tuned hyperparameters
"""

import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ============================================
# 1. Load and Combine Both Datasets
# ============================================

# Dataset 1: spam.csv (UCI SMS Spam Collection)
df1 = pd.read_csv("spam.csv", encoding="latin-1")
df1 = df1[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df1['label'] = df1['label'].map({'ham': 0, 'spam': 1})

# Dataset 2: spam_ham_india.csv
df2 = pd.read_csv("spam_ham_india.csv", encoding="latin-1")
df2 = df2.rename(columns={'Msg': 'message', 'Label': 'label'})
df2['label'] = df2['label'].map({'ham': 0, 'spam': 1})

# Combine
df = pd.concat([df1, df2], ignore_index=True)
df.dropna(subset=['label', 'message'], inplace=True)
df['label'] = df['label'].astype(int)

print(f"Total samples: {len(df)}")
print(f"Spam: {df['label'].sum()}, Ham: {(df['label'] == 0).sum()}")
print(f"Spam ratio: {df['label'].mean() * 100:.1f}%")

# ============================================
# 2. Text Cleaning
# ============================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' url ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['cleaned'] = df['message'].apply(clean_text)

# ============================================
# 3. Feature Extraction (improved TF-IDF)
# ============================================

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),       # unigrams + bigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,        # apply log normalization
    strip_accents='unicode'
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# ============================================
# 4. Train/Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 5. Train Logistic Regression (tuned)
# ============================================

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',  # handle imbalanced data
    solver='lbfgs'
)

model.fit(X_train, y_train)

# ============================================
# 6. Evaluate
# ============================================

y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ============================================
# 7. Quick test on known spam messages
# ============================================

test_messages = [
    "you have won 5000 dollars",
    "Congratulations! You won a free iPhone",
    "WINNER! Claim your prize now",
    "Hey, are you coming to the party tonight?",
    "Free entry to win cash! Text WIN to 80080",
    "Can you pick up some milk on the way home?",
    "URGENT! Your account has been compromised. Click here",
    "Hi mom, I'll be home by 6pm",
]

print("\n--- Quick Test ---")
for msg in test_messages:
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"  [{label}] ({prob[1]*100:.1f}% spam) -> {msg}")

# ============================================
# 8. Save Model & Vectorizer
# ============================================

with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")
