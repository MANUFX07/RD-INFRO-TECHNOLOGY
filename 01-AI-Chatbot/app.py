"""
app.py
Flask backend for the AI-based Customer Support Chatbot.

Serves:
  GET  /            -> chat UI
  POST /chat        -> {"message": "..."} -> {"reply": "...", "intent": "...", "confidence": 0.83}

Real-time NLP: incoming text is cleaned and classified using the TF-IDF +
Logistic Regression pipeline trained in train_model.py. If the model's
confidence is below CONFIDENCE_THRESHOLD, we return the fallback intent's
response instead of guessing — this avoids confidently wrong answers,
which matters for a customer support use case.
"""
import pickle
import random

from flask import Flask, jsonify, render_template, request

from train_model import clean_text

CONFIDENCE_THRESHOLD = 0.35

app = Flask(__name__)

with open("chatbot_model.pkl", "rb") as f:
    saved = pickle.load(f)
    pipeline = saved["pipeline"]
    responses = saved["responses"]


def get_reply(message: str):
    cleaned = clean_text(message)
    if not cleaned:
        tag = "fallback"
        confidence = 0.0
    else:
        proba = pipeline.predict_proba([cleaned])[0]
        classes = pipeline.classes_
        best_idx = proba.argmax()
        tag = classes[best_idx]
        confidence = float(proba[best_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            tag = "fallback"

    reply = random.choice(responses.get(tag, responses["fallback"]))
    return reply, tag, round(confidence, 3)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Please type a message.", "intent": "empty", "confidence": 0}), 400

    reply, intent, confidence = get_reply(message)
    return jsonify({"reply": reply, "intent": intent, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
