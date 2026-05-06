import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "vishnuexe/TamilSense-model"
ID2LABEL = {0: "positive", 1: "negative"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    predicted_id = probs.argmax()
    predicted_label = ID2LABEL[predicted_id]
    confidence = float(probs[predicted_id])

    return {
        "text": text,
        "sentiment": predicted_label,
        "confidence": round(confidence, 4),
        "scores": {
            "positive": round(float(probs[0]), 4),
            "negative": round(float(probs[1]), 4),
        }
    }

if __name__ == "__main__":
    test_sentences = [
        "Super da machan, vera level!",
        "Worst movie ever, total waste of time",
    ]
    for sentence in test_sentences:
        result = predict(sentence)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} ({result['confidence']*100:.1f}% confident)")
        print()