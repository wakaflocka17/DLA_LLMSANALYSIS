import logging
from src import model_factory

# 🧠 Patch temporaneo per usare i repo Hugging Face
from src import model_configs_eval
model_factory.MODEL_CONFIGS = model_configs_eval.MODEL_CONFIGS

# ✅ Carica e valuta l'ensemble
ensemble = model_factory.get_model("ensemble_majority_voting")
ensemble.prepare_datasets(max_samples=500)  # Modifica se vuoi tutti i dati
results = ensemble.evaluate(output_json_path="results/ensemble_eval.json")

logging.info(f"Risultati ensemble:\n{results}")