"""
Post-training evaluation. Produces:
  reports/confusion_matrix.png   - transformer, on the held-out test split
  reports/roc_comparison.png     - baseline vs transformer ROC curves
  reports/calibration.png        - reliability diagram (is 90% confidence really 90%?)
  reports/error_analysis.json    - the most confident mistakes (what fools the model)
  reports/stress_test.json       - hand-written out-of-distribution cases
  reports/summary.json           - everything the README quotes
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).parent
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"
sys.path.insert(0, str(ROOT / "app"))

# Palette (validated default from the dataviz reference): text tokens + two series + one sequential hue
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})

test = pd.read_parquet(DATA / "test.parquet")
y = test.label.values
p_tr = np.load(REPORTS / "test_probs_transformer.npy")
p_bl = joblib.load(MODELS / "tfidf_logreg.joblib").predict_proba(test.content)[:, 1]


def confusion_png():
    cm = confusion_matrix(y, (p_tr >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.grid(False)
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eaf2fc", BLUE]))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{cm[i, j] / cm[i].sum():.1%}", ha="center", va="center",
                    color="#ffffff" if cm[i, j] > cm.max() / 2 else INK, fontsize=12)
    ax.set_xticks([0, 1], ["predicted fake", "predicted real"])
    ax.set_yticks([0, 1], ["actual fake", "actual real"])
    ax.set_title("DistilBERT on 8,117 unseen test articles", loc="left", fontsize=11, color=INK2)
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrix.png", dpi=160); plt.close(fig)


def roc_png():
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for probs, name, color in [(p_bl, "TF-IDF + logistic regression", ORANGE), (p_tr, "Fine-tuned DistilBERT", BLUE)]:
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC {roc_auc_score(y, probs):.4f})")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    ax.set_xlim(0, 0.2); ax.set_ylim(0.8, 1.001)
    ax.set_xlabel("false positive rate (fake called real)"); ax.set_ylabel("true positive rate (real called real)")
    ax.set_title("ROC, zoomed to the top-left corner", loc="left", fontsize=11, color=INK2)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(REPORTS / "roc_comparison.png", dpi=160); plt.close(fig)


def calibration_png():
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--", label="perfectly calibrated")
    for probs, name, color in [(p_bl, "TF-IDF + logistic regression", ORANGE), (p_tr, "Fine-tuned DistilBERT", BLUE)]:
        frac, mean = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=5, lw=2, color=color, label=name)
    ax.set_xlabel("predicted probability of 'real'"); ax.set_ylabel("observed fraction that were real")
    ax.set_title("Reliability diagram (test split)", loc="left", fontsize=11, color=INK2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(REPORTS / "calibration.png", dpi=160); plt.close(fig)


def error_analysis():
    pred = (p_tr >= 0.5).astype(int)
    wrong = test[pred != y].copy()
    wrong["p_real"] = p_tr[pred != y]
    wrong["confidence"] = np.maximum(wrong.p_real, 1 - wrong.p_real)
    wrong = wrong.sort_values("confidence", ascending=False)
    out = [{"true_label": "real" if r.label == 1 else "fake", "p_real": round(float(r.p_real), 4),
            "title": r.title[:160], "excerpt": r.text[:300]} for r in wrong.head(12).itertuples()]
    json.dump({"n_errors": int(len(wrong)), "n_test": int(len(test)), "most_confident_mistakes": out},
              open(REPORTS / "error_analysis.json", "w"), indent=2)
    return len(wrong)


STRESS = [
    # (text, what a human would say, why it is here)
    ("The city council approved a $2.4 million budget for road repairs on Tuesday, council member Maria Lopez said, "
     "adding that work on the downtown stretch of Main Street would begin in the spring.",
     "real", "Local, non-political wire style. Model has seen almost no local news."),
    ("Apple on Tuesday reported quarterly revenue of $94.9 billion, up 6 percent from a year earlier, driven by "
     "iPhone sales, the company said in a statement.",
     "real", "Business news. Dataset is almost entirely US politics."),
    ("A new peer-reviewed study published in The Lancet found that regular walking is associated with a 20 percent "
     "lower risk of cardiovascular disease, researchers at Oxford University reported on Thursday.",
     "real", "Health/science, real in style."),
    ("Researchers at Stanford University reported on Monday that a common food preservative reverses ageing in humans, "
     "the university said in a statement, adding that the finding had been replicated in three independent trials.",
     "fake", "FALSE claim written in perfect wire-service style. The intended failure case: the model checks style, not facts."),
    ("SHOCKING: Doctors HATE this one trick that cures diabetes overnight (VIDEO). Big Pharma is trying to hide this! "
     "Share before it gets deleted. Featured image via YouTube.",
     "fake", "Classic clickbait. Should be easy."),
    ("Scientists confirm the moon landing was staged in a Nevada desert, according to documents leaked to an "
     "independent journalist on Friday. NASA declined to comment.",
     "fake", "Conspiracy content written in neutral tone; no clickbait markers."),
    ("The Federal Reserve held interest rates steady on Wednesday, as expected, and signaled it was in no hurry to "
     "cut, citing inflation that remains above its 2 percent target.",
     "real", "Economic news, wire style."),
    ("Trump Just Did Something So Stupid Even Fox News Couldn't Defend Him (VIDEO)",
     "fake", "Headline only, partisan blog style."),
]


def stress_test():
    from explain import Detector
    det = Detector()
    rows, hits = [], 0
    for text, human, why in STRESS:
        r = det.predict(text)
        ok = r["label"] == human
        hits += ok
        rows.append({"text": text[:120] + ("..." if len(text) > 120 else ""), "human_label": human,
                     "model_label": r["label"], "verdict": r["verdict"], "p_real": r["p_real"], "correct": ok, "why_included": why})
    json.dump({"correct": hits, "total": len(STRESS), "cases": rows}, open(REPORTS / "stress_test.json", "w"), indent=2)
    return hits, len(STRESS)


def main():
    confusion_png(); roc_png(); calibration_png()
    n_err = error_analysis()
    hits, total = stress_test()
    base = json.load(open(REPORTS / "baseline.json"))
    tr = json.load(open(REPORTS / "transformer.json"))
    # accuracy if we abstain below 0.75 confidence (coverage vs accuracy)
    conf = np.maximum(p_tr, 1 - p_tr)
    keep = conf >= 0.75
    acc_kept = float((((p_tr >= 0.5).astype(int) == y)[keep]).mean())
    summary = {
        "test_size": int(len(test)),
        "baseline_clean": base["cleaned"]["metrics"],
        "baseline_raw_leaky": base["raw_with_leakage"]["metrics"],
        "leaky_top_real_features": base["raw_with_leakage"]["top_features"]["pushes_real"][:5],
        "clean_top_real_features": base["cleaned"]["top_features"]["pushes_real"][:5],
        "clean_top_fake_features": base["cleaned"]["top_features"]["pushes_fake"][:5],
        "transformer": tr["test"],
        "transformer_train_minutes": tr["train_minutes"],
        "transformer_errors": n_err,
        "abstain_policy": {"threshold": 0.75, "coverage": round(float(keep.mean()), 4), "accuracy_on_covered": round(acc_kept, 4)},
        "stress_test": {"correct": hits, "total": total},
    }
    json.dump(summary, open(REPORTS / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
