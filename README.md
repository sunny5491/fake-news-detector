# Fake News Detector

Fine-tuned DistilBERT that flags low-credibility news articles, explains its own decision with word-level highlights, and abstains when it isn't confident. Built with a focus on *not* over-claiming what the model can do.

**Live demo (runs in your browser, nothing uploaded):** https://sunny5491.github.io/fake-news-detector/

Local Gradio app: `python3 app/app.py` then open http://127.0.0.1:7860

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
docs/                 static browser demo (GitHub Pages): transformers.js + int8 ONNX + JS port of the baseline
tests/                smoke tests for the detector
deploy/HF_SPACES.md   how to put the Gradio app on Hugging Face Spaces (needs a PRO plan now)
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

Or `./run.sh` for all of the above. Trained weights are not committed (255 MB); they live on the Hub at [Deven5491/distilbert-fakenews](https://huggingface.co/Deven5491/distilbert-fakenews), and `app/explain.py` falls back to that automatically.

## How the browser demo works

Hugging Face stopped offering free Gradio Spaces, so instead of a server the demo runs the model client-side:

- DistilBERT exported to ONNX and dynamically quantised to int8 (268 MB -> 67 MB). On a 600-article test sample this costs 0.3 points of accuracy (98.3% -> 98.0%, 2 flipped verdicts).
- Loaded with [transformers.js](https://github.com/huggingface/transformers.js) from the Hub, cached by the browser after the first visit.
- The TF-IDF + logistic-regression baseline is ported to ~60 lines of JavaScript (`docs/tfidf.js`) and matches scikit-learn to 1e-5. Because it is linear, word-level attributions are exact.
- DistilBERT explanations use sentence-level occlusion: drop each sentence, re-run, report the change in P(real).
