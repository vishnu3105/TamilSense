from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict

app = FastAPI(
    title="Tamil Sentiment Analysis API",
    description="Lightweight Tamil/Tanglish sentiment analysis — positive, negative, or mixed.",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: dict
    response_time_ms: float

@app.get("/")
def root():
    return {"message": "Tamil Sentiment API is live!", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    start = time.time()
    result = predict(request.text)
    response_time = (time.time() - start) * 1000
    return {**result, "response_time_ms": round(response_time, 2)}