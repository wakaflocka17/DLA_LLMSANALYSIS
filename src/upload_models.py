import os
import shutil
import argparse
from utils import upload_model_to_hf

MODELS_TO_UPLOAD = {
    "bert-base-uncased-imdb": "wakaflocka17/bert-imdb-finetuned",
    "bart-base-imdb": "wakaflocka17/bart-imdb-finetuned",
    "gpt-neo-2.7B-imdb": "wakaflocka17/gptneo-imdb-finetuned",
    "majority-voting-imdb": "wakaflocka17/ensemble-majority-voting-imdb"
}

# Directory principali
BASE_MODEL_DIR_FINETUNED = os.path.join(".", "models", "finetuned")
BASE_MODEL_DIR_ENSEMBLE = os.path.join(".", "models", "ensemble")

# Percorsi dei risultati
RESULTS_EVAL_DIR = os.path.join("results", "evaluation", "finetuned")
RESULTS_VALID_DIR = os.path.join("results", "validation", "finetuned")

def copy_results_json(folder_name, model_path):
    eval_subdir = os.path.join(model_path, "evaluation", "finetuned")
    valid_subdir = os.path.join(model_path, "validation", "finetuned")
    os.makedirs(eval_subdir, exist_ok=True)
    os.makedirs(valid_subdir, exist_ok=True)

    eval_file = os.path.join(RESULTS_EVAL_DIR, f"{folder_name}.json")
    valid_file = os.path.join(RESULTS_VALID_DIR, f"{folder_name}_metrics.json")

    if os.path.exists(eval_file):
        shutil.copy(eval_file, os.path.join(eval_subdir, f"{folder_name}.json"))
        print(f"✅ Copiato file di evaluation: {eval_file}")
    else:
        print(f"⚠️ File di evaluation mancante: {eval_file}")

    if os.path.exists(valid_file):
        shutil.copy(valid_file, os.path.join(valid_subdir, f"{folder_name}_metrics.json"))
        print(f"✅ Copiato file di validation: {valid_file}")
    else:
        print(f"⚠️ File di validation mancante: {valid_file}")

def main():
    parser = argparse.ArgumentParser(description="Upload modelli su Hugging Face con metriche")
    parser.add_argument("--only", nargs="*", help="Nome dei modelli da caricare (opzionale)")
    args = parser.parse_args()

    # Se specificato, filtra i modelli
    selected_models = (
        {k: v for k, v in MODELS_TO_UPLOAD.items() if k in args.only}
        if args.only else MODELS_TO_UPLOAD
    )

    for folder_name, hf_repo in selected_models.items():
        if folder_name == "majority-voting-imdb":
            model_path = os.path.join(BASE_MODEL_DIR_ENSEMBLE, folder_name)
        else:
            model_path = os.path.join(BASE_MODEL_DIR_FINETUNED, folder_name)

        if os.path.exists(model_path):
            print(f"🚀 Upload in corso per: {folder_name}")
            copy_results_json(folder_name, model_path)
            upload_model_to_hf(model_path, hf_repo)
        else:
            print(f"❌ ATTENZIONE: Cartella modello non trovata → {model_path}")

if __name__ == "__main__":
    main()