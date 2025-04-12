# upload_models.py

import os
from utils import upload_model_to_hf

# Mappa: cartella locale del modello -> repo Hugging Face
MODELS_TO_UPLOAD = {
    "bert-base-uncased-imdb": "username/bert-imdb-finetuned",
    "bart-base-imdb": "username/bart-imdb-finetuned",
    "gpt-neo-2.7b-imdb": "username/gptneo-imdb-finetuned",
    "ensemble_majority_voting": "username/ensemble-majority-voting-imdb"
}

BASE_MODEL_DIR = "./models"

for folder_name, hf_repo in MODELS_TO_UPLOAD.items():
    model_path = os.path.join(BASE_MODEL_DIR, folder_name)
    if os.path.exists(model_path):
        upload_model_to_hf(model_path, hf_repo)
    else:
        print(f"\u26a0\ufe0f ATTENZIONE: Cartella modello non trovata \u2192 {model_path}")