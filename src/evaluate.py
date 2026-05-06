import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "models/tamil-sentiment-final"
ID2LABEL = {0: "positive", 1: "negative", 2: "mixed"}
LABEL2ID = {"positive": 0, "negative": 1, "mixed": 2}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

df_test = pd.read_csv("data/processed/test.csv")
df_test['label'] = df_test['sentiment'].map(LABEL2ID)

def predict_batch(texts, batch_size=32):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
    return all_preds

print("Running evaluation...")
predictions = predict_batch(df_test['text'].tolist())
true_labels = df_test['label'].tolist()

print("\n=== Classification Report ===")
print(classification_report(
    true_labels,
    predictions,
    target_names=["positive", "negative"]
))

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["positive", "negative"],
    yticklabels=["positive", "negative"]
)
plt.title("Confusion Matrix - MuRIL Tamil Sentiment")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("logs/confusion_matrix.png")
print("\nConfusion matrix saved to logs/confusion_matrix.png")