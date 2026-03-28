import pandas as pd
df = pd.read_csv("data/processed_dataset.csv")
#df = pd.read_csv("data/augmented_dataset.csv")

df = df.dropna(subset=["clean_text"])
df = df[df["clean_text"].astype(str).str.strip() != ""]
df["clean_text"] = df["clean_text"].astype(str)

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

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

# Apply ONLY on training data
augmented_texts = train_texts.apply(synonym_replacement)

# Combine original + augmented training data
train_texts_aug = pd.concat([train_texts, augmented_texts]).reset_index(drop=True)
train_labels_aug = pd.concat([train_labels, train_labels]).reset_index(drop=True)

print(df.head())
print(df["clean_text"].isnull().sum())
print(df["clean_text"].apply(type).value_counts())
print(type(train_texts))
print(len(train_texts))
print(train_texts[:5])

# This will be used for tokenization
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convert to proper format
train_texts = train_texts.astype(str).values.tolist()
test_texts = test_texts.astype(str).values.tolist()

# Tokenization
train_encodings = tokenizer(
    text=train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    text=test_texts,
    truncation=True,
    padding=True,
    max_length=128
)

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

train_dataset = Dataset(train_encodings, train_labels)
test_dataset = Dataset(test_encodings, test_labels)

# We are loading the pre-existing BERT model here 

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training Setup 
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5
)

# Now here we will be training our model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluation 
from sklearn.metrics import accuracy_score, f1_score

predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(axis=1)

print("Accuracy:", accuracy_score(test_labels, preds))
print("F1 Score:", f1_score(test_labels, preds))
