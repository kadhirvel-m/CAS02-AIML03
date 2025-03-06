import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

# Load Pretrained Model and Tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Custom Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {"input_ids": encoding["input_ids"].squeeze(), "attention_mask": encoding["attention_mask"].squeeze(), "labels": torch.tensor(self.labels[idx])}

# Load Dataset (Assuming a CSV file with 'text' and 'label' columns)
data = pd.read_csv("twitter_training.csv")
texts = data["text"].tolist()
labels = data["label"].tolist()

dataset = SentimentDataset(texts, labels)

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train Model
trainer.train()

# Convert Model to TensorFlow and Save as .h5
model_tf = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model_tf.save_pretrained("./fine_tuned_model")
tf.keras.models.save_model(model_tf, "fine_tuned_model.h5")