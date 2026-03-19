import pandas as pd
df = pd.read_csv("data/processed_dataset.csv")

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

print(type(train_texts))
print(type(train_texts.iloc[0]))
print(train_texts.iloc[0])
#print(train_texts.head())
#print(train_texts.isnull().sum())
'''
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet
import random

def synonym_replacement(sentence):
    words = sentence.split()
    new_words = words.copy()

    for i, word in enumerate(words):
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym_words = synonyms[0].lemmas()
            if synonym_words:
                new_words[i] = synonym_words[0].name()

    return " ".join(new_words)

augmented_texts = train_texts.apply(synonym_replacement)

augmented_df = pd.DataFrame({
    "clean_text": augmented_texts,
    "label": train_labels
})

original_df = pd.DataFrame({
    "clean_text": train_texts,
    "label": train_labels
})

augmented_full = pd.concat([original_df, augmented_df])

augmented_full.to_csv("data/augmented_dataset.csv", index=False)
'''