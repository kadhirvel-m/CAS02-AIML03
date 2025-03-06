import pandas as pd
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("twitter_training.csv")

# Preprocess dataset
df = df[['text', 'label']].dropna()

# Map string labels to integers
label_mapping = {"Positive": 1, "Negative": 0, "Neutral": 2}  # Modify based on your dataset
df['label'] = df['label'].map(label_mapping)

df = df.dropna(subset=['label'])  # Drop any rows with NaN labels after mapping
df['label'] = df['label'].astype(int)  # Ensure labels are integers

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def encode_texts(texts, tokenizer, max_length=128):
    return tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors='tf')

# Encode dataset
X = encode_texts(df['text'].tolist(), tokenizer)
y = df['label'].values

import numpy as np
X_train, X_test, y_train, y_test = train_test_split(np.array(X['input_ids']), y, test_size=0.2, random_state=42)


# Convert data to TensorFlow dataset
def convert_to_tf_dataset(X, y, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X)).batch(batch_size)
    return dataset

train_dataset = convert_to_tf_dataset(X_train, y_train)
test_dataset = convert_to_tf_dataset(X_test, y_test)

# Load pre-trained RoBERTa model
model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(set(y)))

# Compile model
model.compile(optimizer=Adam(learning_rate=2e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Train model
model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# Save model as .h5 file
model.save("roberta_model.h5")

print("Model saved as roberta_model.h5")
