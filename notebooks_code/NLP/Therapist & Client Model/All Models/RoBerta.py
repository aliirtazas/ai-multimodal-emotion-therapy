# Importing Packages
import numpy as np
import pandas as pd
import os
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Setting code to run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importing data
segmentation_df = pd.read_parquet(r"C:\Users\sanke\Desktop\Therapist_Model\Segmentation Data\Data\Final Data\Therapy_Session.parquet")
print(segmentation_df.head(10))

# Creating training, testing and validation data
train_df, temp_df = train_test_split(segmentation_df, test_size=0.3, stratify=segmentation_df['speaker'], random_state=310)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['speaker'], random_state=310)
print(f"The shape of the training data: {train_df.shape}")
print(f"The shape of the validation data: {val_df.shape}")
print(f"The shape of the test data: {test_df.shape}")

# Ensure the directory exists
test_data_dir = r"C:\Users\sanke\Desktop\Therapist_Model\Segmentation Data\Data\Final Data"
os.makedirs(test_data_dir, exist_ok=True)

# Save test data only if it does not exist
test_data_path = os.path.join(test_data_dir, "Therapy_Session_Test.parquet")
if not os.path.exists(test_data_path):
    test_df.to_parquet(test_data_path, index=False)
    print(f"Test data saved to {test_data_path}")
else:
    print(f"Test data already exists at {test_data_path}, skipping save.")

# Label Encoding
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["speaker"])
val_df["label"] = label_encoder.transform(val_df["speaker"])

# Tokenization
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_data(examples):
    return tokenizer(examples["utterance"], padding="max_length", truncation=True, max_length=512)

# Convert DataFrames to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['utterance', 'label']])
val_dataset = Dataset.from_pandas(val_df[['utterance', 'label']])

# Apply tokenization separately
train_dataset = train_dataset.map(tokenize_data, batched=True)
val_dataset = val_dataset.map(tokenize_data, batched=True)

# Ensure correct format
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

# Model setup
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_encoder.classes_)).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted')
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# Save model, tokenizer and label encoder
saved_model_dir = r"C:\Users\sanke\Desktop\Therapist_Model\Saved Model"
os.makedirs(saved_model_dir, exist_ok=True)
model.save_pretrained(saved_model_dir)
tokenizer.save_pretrained(saved_model_dir)
joblib.dump(label_encoder, os.path.join(saved_model_dir, "label_encoder.pkl"))
print(f"Model, tokenizer and label encoder saved to {saved_model_dir}")
