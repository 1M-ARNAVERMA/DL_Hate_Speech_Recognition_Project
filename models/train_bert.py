import pandas as pd
df = pd.read_csv("data/processed_dataset.csv")

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42
)