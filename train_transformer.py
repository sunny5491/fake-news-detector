"""
Fine-tune DistilBERT for fake-vs-real news classification on Apple Silicon (MPS).

Custom training loop (rather than HF Trainer) so it stays inside 8 GB of RAM:
- max sequence length 256 tokens (title + first ~200 words is where the signal is)
- batch size 16, gradient clipping, linear warmup/decay
- evaluation on validation each epoch, final metrics on the held-out test split
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).parent
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"
OUT = MODELS / "distilbert-fakenews"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH = 16
EPOCHS = 1
LR = 3e-5
TRAIN_SAMPLES = None  # None = full train split; set an int to subsample for a quick run

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)


class NewsDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=False)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i], "labels": self.labels[i]}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        mask[i, :n] = 1
    return ids, mask, torch.tensor([b["labels"] for b in batch])


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, labels = [], []
    for ids, mask, y in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(probs, y):
    pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision_real": round(precision_score(y, pred), 4),
        "recall_real": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, probs), 4),
        "confusion": {"tn_fake_as_fake": int(tn), "fp_fake_as_real": int(fp), "fn_real_as_fake": int(fn), "tp_real_as_real": int(tp)},
    }


def main():
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train = pd.read_parquet(DATA / "train.parquet")
    val = pd.read_parquet(DATA / "validation.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    if TRAIN_SAMPLES:
        train = train.sample(TRAIN_SAMPLES, random_state=42)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    col = lambda b: collate(b, tok.pad_token_id)
    tr_loader = DataLoader(NewsDS(train.content, train.label, tok), batch_size=BATCH, shuffle=True, collate_fn=col)
    va_loader = DataLoader(NewsDS(val.content, val.label, tok), batch_size=64, collate_fn=col)
    te_loader = DataLoader(NewsDS(test.content, test.label, tok), batch_size=64, collate_fn=col)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "fake", 1: "real"}, label2id={"fake": 0, "real": 1}
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = EPOCHS * len(tr_loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    history, step, t0 = [], 0, time.time()
    for ep in range(EPOCHS):
        model.train()
        running = 0.0
        for ids, mask, y in tr_loader:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            running += out.loss.item(); step += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"step {step}/{total}  loss {running/50:.4f}  {el/60:.1f} min elapsed, ~{el/step*(total-step)/60:.1f} min left", flush=True)
                running = 0.0
        vp, vy = predict(model, va_loader)
        vm = metrics(vp, vy)
        print(f"epoch {ep+1} validation:", vm, flush=True)
        history.append({"epoch": ep + 1, "validation": vm})

    tp_, ty = predict(model, te_loader)
    tm = metrics(tp_, ty)
    print("TEST:", tm, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    np.save(REPORTS / "test_probs_transformer.npy", tp_)
    json.dump(
        {"model": MODEL_NAME, "max_len": MAX_LEN, "batch": BATCH, "epochs": EPOCHS, "lr": LR,
         "train_samples": len(train), "train_minutes": round((time.time() - t0) / 60, 1),
         "history": history, "test": tm},
        open(REPORTS / "transformer.json", "w"), indent=2,
    )
    print("saved to", OUT)


if __name__ == "__main__":
    main()
