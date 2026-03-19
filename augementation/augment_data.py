import pandas as pd

df = pd.read_csv("data/processed_dataset.csv")

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# IMPORTANT: reset index
train_texts = train_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)

# ---- AUGMENTATION STARTS ----
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
                new_word = synonym_words[0].name()
                new_word = new_word.replace("_", " ")   
                new_words[i] = new_word

    return " ".join(new_words)

# Apply augmentation
augmented_texts = train_texts.apply(synonym_replacement)

# Create datasets
augmented_df = pd.DataFrame({
    "clean_text": augmented_texts,
    "label": train_labels
})

original_df = pd.DataFrame({
    "clean_text": train_texts,
    "label": train_labels
})


# Combine
augmented_full = pd.concat([original_df, augmented_df]).reset_index(drop=True)

augmented_full["clean_text"] = augmented_full["clean_text"].astype(str)

# Remove empty or very short sentences
augmented_full = augmented_full[augmented_full["clean_text"].str.strip() != ""]
augmented_full = augmented_full[augmented_full["clean_text"].str.len() > 3]
# Save
augmented_full.to_csv("data/augmented_dataset.csv", index=False)

print("Augmented dataset created successfully!")