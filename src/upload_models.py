import os
from utils import upload_model_to_hf

# Mappa: nome della cartella locale -> repo Hugging Face
# Le chiavi devono corrispondere ai nomi usati nelle cartelle
MODELS_TO_UPLOAD = {
    "bert-base-uncased-imdb": "wakaflocka17/bert-imdb-finetuned",
    "bart-base-imdb": "wakaflocka17/bart-imdb-finetuned",
    "gpt-neo-2.7B-imdb": "wakaflocka17/gptneo-imdb-finetuned",
    "majority-voting-imdb": "wakaflocka17/ensemble-majority-voting-imdb"
}

# Definiamo le directory base per i modelli fine-tunati e per l'ensemble
BASE_MODEL_DIR_FINETUNED = os.path.join(".", "models", "finetuned")
BASE_MODEL_DIR_ENSEMBLE = os.path.join(".", "models", "ensemble")

for folder_name, hf_repo in MODELS_TO_UPLOAD.items():
    # Se il nome della cartella indica l'ensemble, usiamo la directory ensemble,
    # altrimenti assumiamo che si tratti di un modello fine-tunato.
    if folder_name == "majority-voting-imdb":
        model_path = os.path.join(BASE_MODEL_DIR_ENSEMBLE, folder_name)
    else:
        model_path = os.path.join(BASE_MODEL_DIR_FINETUNED, folder_name)
    
    if os.path.exists(model_path):
        upload_model_to_hf(model_path, hf_repo)
    else:
        print(f"\u26a0\ufe0f ATTENZIONE: Cartella modello non trovata \u2192 {model_path}")