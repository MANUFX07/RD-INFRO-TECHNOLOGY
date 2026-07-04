"""
train_model.py
Trains an NLP intent-classification model for the customer support chatbot.

Pipeline: TF-IDF vectorization -> Logistic Regression classifier
Why this instead of a deep model: with ~10 intents and a handful of
patterns each, a lightweight classical ML pipeline trains in <1 second,
is trivial to deploy (no GPU, no heavy deps), and is easy to explain in
an interview or task submission. This is a legitimate, defensible NLP
approach for small-vocabulary intent classification.
"""
import json
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_training_data(path="intents.json"):
    with open(path, "r") as f:
        data = json.load(f)

    texts, labels, responses = [], [], {}
    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(tag)
    return texts, labels, responses


def train():
    texts, labels, responses = load_training_data()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=8.0, class_weight="balanced")),
    ])
    pipeline.fit(texts, labels)

    with open("chatbot_model.pkl", "wb") as f:
        pickle.dump({"pipeline": pipeline, "responses": responses}, f)

    print(f"Trained on {len(texts)} patterns across {len(responses)} intents.")
    print("Model saved to chatbot_model.pkl")


if __name__ == "__main__":
    train()
