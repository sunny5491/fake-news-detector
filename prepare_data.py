"""
Download the fake-news dataset and clean it.

Why the cleaning matters (responsible-AI point #1):
The underlying ISOT corpus collected its REAL articles from Reuters and its
FAKE articles from flagged websites. Almost every real article therefore starts
with a dateline like "WASHINGTON (Reuters) -". A model can "cheat" by learning
that single token instead of learning anything about the content. We strip
those source fingerprints so the reported accuracy reflects the language of the
article, not the name of the publisher.
"""
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# Dateline pattern: "WASHINGTON (Reuters) - " or "(Reuters) -"
DATELINE = re.compile(r"^[^\n]{0,80}?\(Reuters\)\s*-\s*", re.IGNORECASE)
SOURCE_WORDS = re.compile(
    r"\b(reuters|21st century wire|breitbart|infowars|the onion)\b", re.IGNORECASE
)
URL = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    text = text or ""
    text = DATELINE.sub("", text)
    text = SOURCE_WORDS.sub("", text)
    text = URL.sub("", text)
    return WHITESPACE.sub(" ", text).strip()


def main():
    ds = load_dataset("GonzaloA/fake_news")
    for split in ["train", "validation", "test"]:
        df = ds[split].to_pandas()
        df = df.rename(columns={"Unnamed: 0": "id"})
        df["title"] = df["title"].fillna("").map(clean)
        df["text"] = df["text"].fillna("").map(clean)
        df = df[(df["text"].str.len() > 20) | (df["title"].str.len() > 5)]
        df = df.drop_duplicates(subset=["title", "text"])
        # Combined input the models will see
        df["content"] = (df["title"] + ". " + df["text"]).str.strip()
        df = df[["id", "title", "text", "content", "label"]].reset_index(drop=True)
        df.to_parquet(OUT / f"{split}.parquet", index=False)
        print(f"{split:11s} rows={len(df):6d}  fake={int((df.label==0).sum()):6d}  real={int((df.label==1).sum()):6d}")

    # Label semantics in this dataset: 0 = fake, 1 = real
    (OUT / "LABELS.txt").write_text("0 = fake\n1 = real\n")


if __name__ == "__main__":
    main()
