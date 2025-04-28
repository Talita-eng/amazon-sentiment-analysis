import re
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

def clean_dataset(df, text_column):
    cleaner = TextPreprocessor()
    df[text_column] = df[text_column].astype(str).apply(cleaner.clean_text)
    return df
