# Fake News Detector

Fine-tuned DistilBERT that flags low-credibility news articles, explains its own decision with word-level highlights, and abstains when it isn't confident. Built with a focus on *not* over-claiming what the model can do.

**Live demo:** run `python3 app/app.py` and open http://127.0.0.1:7860

## Results (held-out test set, 8,117 articles)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.973 | 0.975 | 0.995 |
| DistilBERT (1 epoch, 39 min on M-series MPS) | **0.985** | **0.986** | **0.999** |

With the abstain policy (refuse below 75% confidence) the model answers 97.4% of articles and is 99.5% accurate on those.

## What's in here

```
prepare_data.py       download the ISOT-style dataset, strip source fingerprints, split
train_baseline.py     TF-IDF + logreg baseline, plus a leakage experiment
train_transformer.py  fine-tune distilbert-base-uncased (MPS / CPU)
eval_report.py        ROC, calibration, confusion matrix, error analysis, stress test
app/explain.py        inference + gradient x input word attributions + abstain
app/app.py            Gradio UI
tests/                smoke tests for the detector
deploy/HF_SPACES.md   how to put it on Hugging Face Spaces for free
run.sh                reproduce everything end to end
```

## Three things I did on purpose

**1. Removed the label leak.** In this dataset almost every *real* article starts with `WASHINGTON (Reuters) -` and the fake ones never do. A model can score 97% by learning one token. `prepare_data.py` strips datelines, publisher names and URLs so the numbers above reflect the writing, not the byline. The leaky version scores slightly *higher* (0.976 vs 0.973), which is exactly why it's misleading.

**2. Made it explain itself.** Every prediction comes with signed word attributions (gradient x input on the embeddings, merged from word-pieces back to words). Green pushed toward *real*, red toward *fake*. Attributed quotes and datelines read as real; `VIDEO`, `featured image`, `read more`, ALL-CAPS outrage read as fake.

**3. Let it say "I don't know".** Below 75% confidence the app returns *uncertain* instead of a verdict, and it cross-checks against the TF-IDF baseline and flags disagreement.

## What it can't do

It learns **writing style**, not truth. A false claim written in neutral wire-service prose gets called *real* with high confidence, and the stress test in `reports/stress_test.json` shows exactly that (6/8, both misses are this case). Training data is 2016-2017 US politics, so it's out of distribution on local news, business, science and anything after 2017. Treat it as a triage aid, never a judge.

## Run it

```bash
pip3 install -r requirements.txt
python3 prepare_data.py
python3 train_baseline.py
python3 train_transformer.py     # ~40 min on an M-series Mac
python3 eval_report.py
python3 -m pytest tests -q
python3 app/app.py
```

Or `./run.sh` for all of the above. Trained weights are not committed (255 MB); see `deploy/HF_SPACES.md` for pushing them to the Hub.
