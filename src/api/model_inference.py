# src/api/model_inference.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model (same one you used in your pipeline)
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

labels = ["negative", "neutral", "positive"]

def predict_finbert_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_idx = torch.argmax(probs, dim=1).item()
        sentiment_label = labels[sentiment_idx]
        sentiment_score = probs[0][sentiment_idx].item()
    return {
        "input_text": text,
        "sentiment_score": round(sentiment_score, 3),
        "sentiment_label": sentiment_label.capitalize()
    }
