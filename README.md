---
title: TamilSense
emoji: 🌟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.10"
app_file: app/gradio_app.py
pinned: false
---

# TamilSense 🌟
> Lightweight, production-grade Tamil/Tanglish sentiment analysis — built for real-world deployment

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-TamilSense-blue)](https://huggingface.co/spaces/vishnuexe/TamilSense)
[![Model](https://img.shields.io/badge/🤗%20Model-TamilSense--model-green)](https://huggingface.co/vishnuexe/TamilSense-model)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 What Is This?

ChatGPT is a trillion-parameter monster that costs millions to run. Nobody is integrating it into a local government portal, a small business app, or a low-bandwidth mobile app in rural Tamil Nadu.

**TamilSense fills that gap** — a lightweight, fast, open-source Tamil sentiment API that any developer can plug in.

## 📊 Performance

| Metric | Score |
|--------|-------|
| Accuracy | **94.7%** |
| Weighted F1 | **94.7%** |
| Positive F1 | **97%** |
| Negative F1 | **85%** |

Trained on **55,064 balanced Tamil-English sentences** from two combined datasets.

## 🚀 Live Demo

👉 [**Try it on Hugging Face Spaces**](https://huggingface.co/spaces/vishnuexe/TamilSense)

## ⚡ Quick Start

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="vishnuexe/TamilSense-model")
result = classifier("Super da machan vera level!")
print(result)  # [{'label': 'positive', 'score': 0.9965}]
```

## 🛠️ REST API

```bash
# Run locally
uvicorn app.main:app --reload

# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Romba nalla iruku bro"}'
```

Response:
```json
{
  "text": "Romba nalla iruku bro",
  "sentiment": "positive",
  "confidence": 0.9966,
  "scores": {"positive": 0.9966, "negative": 0.0034},
  "response_time_ms": 48.23
}
```

## 🏗️ Architecture
Tamil/Tanglish Text
↓
MuRIL Tokenizer (WordPiece)
↓
12 Transformer Layers (Google MuRIL)
↓
[CLS] Vector (768-dim)
↓
Linear Classifier → Positive / Negative

## 📦 Dataset

| Source | Size |
|--------|------|
| tamilmixsentiment (FIRE 2020) | 15,744 sentences |
| DravidianCodeMix Zenodo | ~44,000 sentences |
| **Final balanced train set** | **55,064 sentences** |

## 🔬 MLOps Pipeline

- **Experiment tracking**: MLflow — 4 training runs logged with params and metrics
- **GPU training**: NVIDIA RTX 4060 — fp16 mixed precision, ~45 mins
- **Model hosting**: Hugging Face Model Hub
- **Deployment**: Hugging Face Spaces (Gradio) + FastAPI
- **Containerization**: Docker

## 📈 Training Progress

| Run | Data | F1 |
|-----|------|----|
| Run 1 — 3-class | 10k sentences | 67.5% |
| Run 2 — Binary | 15k sentences | 80.8% |
| Run 3 — Tuned | 15k sentences | 81.5% |
| **Run 4 — Full data** | **55k sentences** | **94.7%** |

## 🗂️ Project Structure
TamilSense/
├── app/
│   ├── main.py          # FastAPI REST API
│   └── gradio_app.py    # Gradio demo UI
├── src/
│   ├── prepare_data.py  # Data loading and balancing
│   ├── train.py         # Fine-tuning with MLflow tracking
│   ├── evaluate.py      # Evaluation + confusion matrix
│   └── predict.py       # Inference pipeline
├── Dockerfile
└── requirements.txt

## 🔮 Roadmap

- [ ] IndicBERT experiment on pure Tamil script dataset
- [ ] INT8 quantization — reduce model size 4x
- [ ] Batch prediction API endpoint
- [ ] FIRE 2021/2022 data integration
- [ ] Offensive language detection endpoint (v2)

## 📄 License

MIT License — free to use, modify, and deploy.

---

**Built by Vishnu** | [HuggingFace](https://huggingface.co/vishnuexe) | [GitHub](https://github.com/vishnu3105)
