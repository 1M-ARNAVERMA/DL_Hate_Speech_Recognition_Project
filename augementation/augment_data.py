import pandas as pd
import nltk
nltk.download('wordnet')
df = pd.read_csv("data/processed_dataset.csv")

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

augmented_full = pd.concat([
    pd.DataFrame({"clean_text": train_texts, "label": train_labels}),
    augmented_df
])
