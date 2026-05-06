import mlflow
import mlflow.pytorch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

MODEL_NAME = "google/muril-base-cased"
NUM_LABELS = 2
MAX_LENGTH = 128
LABEL2ID = {"positive": 0, "negative": 1}
ID2LABEL = {0: "positive", 1: "negative"}

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

df_train = pd.read_csv('data/processed/train.csv')
df_test = pd.read_csv('data/processed/test.csv')

df_train['label'] = df_train['sentiment'].map(LABEL2ID)
df_test['label'] = df_test['sentiment'].map(LABEL2ID)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
test_dataset = Dataset.from_pandas(df_test[['text', 'label']])
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="models/tamil-sentiment",
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    logging_dir="logs",
    logging_steps=50,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=1e-5,
)

mlflow.set_tracking_uri("file:///D:/tamil/mlruns")
mlflow.set_experiment("tamil-sentiment")

with mlflow.start_run(run_name="muril-binary-v4"):
    mlflow.log_param("train_size", 55064)
    mlflow.log_param("model", MODEL_NAME)
    mlflow.log_param("epochs", 8)
    mlflow.log_param("max_length", MAX_LENGTH)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("learning_rate", 1e-5)
    mlflow.log_param("classes", "binary")
    mlflow.log_param("train_size", len(df_train))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()
    mlflow.log_metric("accuracy", results['eval_accuracy'])
    mlflow.log_metric("f1", results['eval_f1'])

    print(f"\nAccuracy: {results['eval_accuracy']:.4f}")
    print(f"F1 Score: {results['eval_f1']:.4f}")

    trainer.save_model("models/tamil-sentiment-final")
    tokenizer.save_pretrained("models/tamil-sentiment-final")
    print("Model saved!")