"""
Inference + explainability for the fine-tuned DistilBERT.

Word attributions use gradient x input on the token embeddings, merged from
word-pieces back into whole words. Positive score = pushed toward REAL, negative
= pushed toward FAKE. This is a cheap, honest approximation (one backward
pass); it shows what the model attended to, not whether a claim is true.
"""
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TRANSFORMER_DIR = ROOT / "models" / "distilbert-fakenews"
BASELINE_PATH = ROOT / "models" / "tfidf_logreg.joblib"
MAX_LEN = 256

# Responsible-AI guardrail: below this confidence we refuse to give a verdict.
ABSTAIN_BELOW = 0.75

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Detector:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR).to(device).eval()
        self.baseline = joblib.load(BASELINE_PATH) if BASELINE_PATH.exists() else None

    def predict(self, text: str) -> dict:
        text = (text or "").strip()
        if len(text.split()) < 5:
            return {"error": "Please paste at least a full sentence (5+ words)."}

        enc = self.tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

        emb_layer = self.model.get_input_embeddings()
        embeds = emb_layer(ids).detach().requires_grad_(True)
        out = self.model(inputs_embeds=embeds, attention_mask=mask)
        probs = torch.softmax(out.logits.float(), dim=-1)[0]
        p_real = float(probs[1].detach())

        # gradient of the REAL logit minus FAKE logit -> signed attribution
        score = out.logits[0, 1] - out.logits[0, 0]
        score.backward()
        attr = (embeds.grad * embeds).sum(-1)[0].detach().cpu().numpy()

        words = self._merge_wordpieces(ids[0].cpu().tolist(), attr)
        truncated = len(self.tok(text)["input_ids"]) > MAX_LEN

        label = "real" if p_real >= 0.5 else "fake"
        confidence = max(p_real, 1 - p_real)
        verdict = label if confidence >= ABSTAIN_BELOW else "uncertain"

        result = {
            "verdict": verdict,
            "label": label,
            "confidence": round(confidence, 4),
            "p_real": round(p_real, 4),
            "p_fake": round(1 - p_real, 4),
            "words": words,
            "truncated": truncated,
        }
        if self.baseline is not None:
            b = float(self.baseline.predict_proba([text])[0, 1])
            result["baseline_p_real"] = round(b, 4)
            result["models_agree"] = (b >= 0.5) == (p_real >= 0.5)
        return result

    def _merge_wordpieces(self, ids, attr):
        toks = self.tok.convert_ids_to_tokens(ids)
        words, cur, cur_score = [], "", 0.0
        for t, a in zip(toks, attr):
            if t in self.tok.all_special_tokens:
                continue
            if t.startswith("##"):
                cur += t[2:]; cur_score += a
            else:
                if cur:
                    words.append((cur, cur_score))
                cur, cur_score = t, a
        if cur:
            words.append((cur, cur_score))
        if not words:
            return []
        m = max(abs(s) for _, s in words) or 1.0
        return [(w, round(float(s / m), 3)) for w, s in words]  # normalized to [-1, 1]
