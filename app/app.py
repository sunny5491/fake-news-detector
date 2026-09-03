"""Gradio UI for the fake-news detector. Run: python3 app/app.py"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explain import ABSTAIN_BELOW, Detector  # noqa: E402

det = Detector()

DISCLAIMER = (
    "**What this model actually does.** It was trained on ~24k US political news articles (2016-2017) "
    "labelled by *source reputation*, not by fact-checking individual claims. It learns the *writing style* of "
    "wire-service journalism versus partisan blogs: attributed quotes and datelines push toward *real*; "
    "'VIDEO', 'featured image', ALL-CAPS outrage push toward *fake*. "
    "It does **not** verify facts, and a well-written false story can fool it. "
    f"Below {int(ABSTAIN_BELOW*100)}% confidence it abstains rather than guess. Treat it as a triage aid, never a judge."
)

EXAMPLES = [
    ["The Senate Judiciary Committee voted 12-10 on Thursday to advance the nomination, sending it to the full Senate, "
     "where a vote is expected next week, committee chairman Chuck Grassley said in a statement."],
    ["BREAKING: You Won't BELIEVE What This Senator Was Just Caught Doing (VIDEO). Share this before they take it down! "
     "Featured image via screenshot. Read more at the link below."],
    ["Scientists at the university published a study in Nature on Wednesday reporting that the new compound reduced tumor "
     "growth in mice by 40 percent, though they cautioned that human trials are still years away."],
]


def run(text):
    r = det.predict(text)
    if "error" in r:
        return r["error"], None, [], ""

    if r["verdict"] == "uncertain":
        headline = f"### ⚠️ Uncertain ({r['confidence']*100:.1f}% toward *{r['label']}*) — not confident enough to call it."
    elif r["verdict"] == "real":
        headline = f"### ✅ Looks like legitimate news reporting — {r['confidence']*100:.1f}% confidence"
    else:
        headline = f"### 🚩 Looks like fake / low-credibility content — {r['confidence']*100:.1f}% confidence"

    probs = {"real": r["p_real"], "fake": r["p_fake"]}
    highlighted = [(w + " ", s) for w, s in r["words"]]

    notes = []
    if "models_agree" in r:
        agree = "agree" if r["models_agree"] else "**disagree**"
        notes.append(f"Cross-check: the TF-IDF baseline says {r['baseline_p_real']*100:.0f}% real, so the two models {agree}.")
    if r["truncated"]:
        notes.append("Only the first ~200 words were analysed (model input limit).")
    notes.append("Highlight colours: green words pushed toward *real*, red toward *fake*. Intensity = strength.")
    return headline, probs, highlighted, "\n\n".join(notes)


with gr.Blocks(title="Fake News Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📰 Fake News Detector\nFine-tuned DistilBERT · explains its own decision · abstains when unsure")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(lines=10, label="Paste a news headline + article text", placeholder="Paste article here...")
            btn = gr.Button("Analyse", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp, label="Try an example")
        with gr.Column(scale=2):
            head = gr.Markdown()
            lab = gr.Label(num_top_classes=2, label="Probabilities")
            notes = gr.Markdown()
    hl = gr.HighlightedText(label="Why? (word-level attribution)", combine_adjacent=False, show_legend=False,
                            color_map=None)
    gr.Markdown(DISCLAIMER)
    btn.click(run, inputs=inp, outputs=[head, lab, hl, notes])
    inp.submit(run, inputs=inp, outputs=[head, lab, hl, notes])

if __name__ == "__main__":
    demo.launch()
