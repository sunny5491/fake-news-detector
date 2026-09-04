# Deploying to Hugging Face Spaces (free, public link for your resume)

1. Create a Space at https://huggingface.co/new-space -> SDK: **Gradio**, hardware: CPU basic (free).
2. Push the trained model to the Hub so the Space can download it:
   ```bash
   pip3 install -U "huggingface-hub<1.0"
   huggingface-cli login
   huggingface-cli upload <your-username>/distilbert-fakenews models/distilbert-fakenews .
   huggingface-cli upload <your-username>/distilbert-fakenews models/tfidf_logreg.joblib tfidf_logreg.joblib
   ```
3. In `app/explain.py`, set `TRANSFORMER_DIR = "<your-username>/distilbert-fakenews"` and download
   the joblib with `hf_hub_download` (the Space has no local `models/` folder).
4. Copy `app/app.py`, `app/explain.py`, and `requirements.txt` into the Space repo; Spaces run `app.py` at the root,
   so either flatten the two files or add `app_file: app/app.py` to the Space README front-matter.
5. Push. The Space builds in ~5 minutes and gives you `https://huggingface.co/spaces/<you>/fake-news-detector`.
