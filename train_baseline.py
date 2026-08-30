"""
Baseline: TF-IDF + Logistic Regression.

Two purposes:
1. A fast, interpretable reference point for the transformer.
2. A leakage experiment (responsible-AI point #2): we train the same model on the
   RAW text (publisher datelines intact) and on the CLEANED text, then look at
   the most influential words. If "reuters" is the top feature on raw text, the
   dataset has a shortcut the model can exploit.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, max_iter=2000)),
    ])


def evaluate(pipe, X, y):
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision_real": round(precision_score(y, pred), 4),
        "recall_real": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def top_features(pipe, k=15):
    vocab = np.array(pipe["tfidf"].get_feature_names_out())
    coef = pipe["clf"].coef_[0]
    order = np.argsort(coef)
    return {"pushes_real": vocab[order[-k:]][::-1].tolist(), "pushes_fake": vocab[order[:k]].tolist()}


def main():
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")

    # ---- Cleaned text (the model we ship) ----
    pipe = make_pipeline().fit(train.content, train.label)
    clean_metrics = evaluate(pipe, test.content, test.label)
    clean_feats = top_features(pipe)
    joblib.dump(pipe, MODELS / "tfidf_logreg.joblib")
    print("CLEAN  :", clean_metrics)
    print("  real <-", clean_feats["pushes_real"][:10])
    print("  fake <-", clean_feats["pushes_fake"][:10])

    # ---- Raw text (leakage experiment, not shipped) ----
    raw = load_dataset("GonzaloA/fake_news")
    rtr, rte = raw["train"].to_pandas(), raw["test"].to_pandas()
    for d in (rtr, rte):
        d["content"] = (d["title"].fillna("") + ". " + d["text"].fillna(""))
    raw_pipe = make_pipeline().fit(rtr.content, rtr.label)
    raw_metrics = evaluate(raw_pipe, rte.content, rte.label)
    raw_feats = top_features(raw_pipe)
    print("RAW    :", raw_metrics)
    print("  real <-", raw_feats["pushes_real"][:10])
    print("  fake <-", raw_feats["pushes_fake"][:10])

    json.dump(
        {"cleaned": {"metrics": clean_metrics, "top_features": clean_feats},
         "raw_with_leakage": {"metrics": raw_metrics, "top_features": raw_feats}},
        open(REPORTS / "baseline.json", "w"), indent=2,
    )


if __name__ == "__main__":
    main()
