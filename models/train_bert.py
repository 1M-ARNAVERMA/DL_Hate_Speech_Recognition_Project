import pandas as pd
import numpy as np

# Load ORIGINAL dataset (correct)
df = pd.read_csv("data/processed_dataset.csv")

# Basic cleaning
df = df.dropna(subset=["clean_text"])
df = df[df["clean_text"].astype(str).str.strip() != ""]
df["clean_text"] = df["clean_text"].astype(str)

# Train-test split FIRST (important)
from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Reset indices
train_texts = train_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
test_texts = test_texts.reset_index(drop=True)
test_labels = test_labels.reset_index(drop=True)

# -------------------------------
# DATA AUGMENTATION (TRAIN ONLY)
# -------------------------------
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet
import random

def synonym_replacement(sentence):
    words = sentence.split()
    new_words = words.copy()

    if len(words) > 0:
        i = random.randint(0, len(words)-1)
        synonyms = wordnet.synsets(words[i])
        if synonyms:
            synonym_words = synonyms[0].lemmas()
            if synonym_words:
                new_word = synonym_words[0].name().replace("_", " ")
                new_words[i] = new_word

    return " ".join(new_words)

# Apply augmentation ONLY on training data
augmented_texts = train_texts.apply(synonym_replacement)

# Combine original + augmented
train_texts_aug = pd.concat([train_texts, augmented_texts]).reset_index(drop=True)
train_labels_aug = pd.concat([train_labels, train_labels]).reset_index(drop=True)

# -------------------------------
# TOKENIZATION
# -------------------------------
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convert to list format
train_texts_list = train_texts.astype(str).values.tolist()
test_texts_list = test_texts.astype(str).values.tolist()
train_texts_aug_list = train_texts_aug.astype(str).values.tolist()
train_labels_aug_list = train_labels_aug.tolist()

# Tokenize
train_encodings = tokenizer(
    text=train_texts_list,
    truncation=True,
    padding=True,
    max_length=128
)

train_encodings_aug = tokenizer(
    text=train_texts_aug_list,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    text=test_texts_list,
    truncation=True,
    padding=True,
    max_length=128
)

# -------------------------------
# DATASET CLASS
# -------------------------------
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = Dataset(train_encodings, train_labels)
train_dataset_aug = Dataset(train_encodings_aug, train_labels_aug)
test_dataset = Dataset(test_encodings, test_labels)

# -------------------------------
# MODEL + TRAINING SETUP
# -------------------------------
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5
)

# -------------------------------
# BASELINE MODEL
# -------------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluate baseline
from sklearn.metrics import accuracy_score, f1_score

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

baseline_acc = accuracy_score(test_labels, preds)
baseline_f1 = f1_score(test_labels, preds)

print("Baseline Accuracy:", baseline_acc)
print("Baseline F1:", baseline_f1)

# -------------------------------
# AUGMENTED MODEL
# -------------------------------
model_aug = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

trainer_aug = Trainer(
    model=model_aug,
    args=training_args,
    train_dataset=train_dataset_aug,
    eval_dataset=test_dataset
)

trainer_aug.train()

# Evaluate augmented
predictions_aug = trainer_aug.predict(test_dataset)
preds_aug = np.argmax(predictions_aug.predictions, axis=1)

aug_acc = accuracy_score(test_labels, preds_aug)
aug_f1 = f1_score(test_labels, preds_aug)

print("Augmented Accuracy:", aug_acc)
print("Augmented F1:", aug_f1)

# -------------------------------
# FINAL COMPARISON
# -------------------------------
print("\n=== FINAL RESULTS ===")
print(f"Baseline  -> Accuracy: {baseline_acc:.4f}, F1: {baseline_f1:.4f}")
print(f"Augmented -> Accuracy: {aug_acc:.4f}, F1: {aug_f1:.4f}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Baseline")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.show()