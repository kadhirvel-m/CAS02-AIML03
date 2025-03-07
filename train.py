import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from tqdm import tqdm
import os

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Label mapping
label_mapping = {"Positive": 1, "Negative": 0, "Neutral": 2}  # Adjust as needed

# Load dataset
class SentimentDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_file)

        # Ensure labels are mapped correctly
        self.data['label'] = self.data['label'].map(label_mapping)
        self.data = self.data.dropna(subset=['label'])  # Remove any NaN labels
        self.data = self.data[self.data['label'].isin([0, 1, 2])]  # Keep valid labels

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row['text']) if pd.notna(row['text']) else ""  # Ensure text is string
        label = int(row['label'])  # Convert label to integer

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)  # Ensure integer label
        }

# Load dataset
dataset = SentimentDataset("Training.csv", tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

print("Training complete!")

# Save the trained model in .pt format
output_dir = "saved_model"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "roberta_sentiment.pt")
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
