
import pandas as pd
import string
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word not in stop_words])

def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['cleaned'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def train_model(df, vectorizer):
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    return model

def evaluate_model(y_true, y_pred):
    print("\n📊 Accuracy:", accuracy_score(y_true, y_pred))
    print("\n📑 Classification Report:\n", classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def predict_message(model, vectorizer, msg):
    cleaned = clean_text(msg)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    label = "SPAM 🚫" if prediction == 1 else "HAM ✅"
    print(f"\n📨 Message: {msg}")
    print(f"📢 Prediction: {label}")

def save_model(model, vectorizer):
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("\n✅ Model and vectorizer saved!")

if __name__ == "__main__":
    df = load_data()
    vectorizer = CountVectorizer()
    model = train_model(df, vectorizer)
    save_model(model, vectorizer)

    test_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/12345 now.",
        "Hey, are we still on for lunch today?",
        "URGENT! Your account has been suspended. Reply with your password to reactivate.",
        "Can you send me the report by tomorrow?",
        "You have been selected for a free cruise. Call now!"
    ]

    print("\n🔍 Testing sample messages...")
    for msg in test_messages:
        predict_message(model, vectorizer, msg)
