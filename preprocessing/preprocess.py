import re

df = pd.read_csv("data/labeled_data.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"#", "", text)            # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters
    return text

banned_words = [
    "word1",
    "word2",
    "word3"
]

def contains_banned_words(text):
    return any(word in text for word in banned_words)

df = df[~df["clean_text"].apply(contains_banned_words)]