from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import os

# Mappa: nome logico -> repo Hugging Face
models = {
    "bert_base_uncased": "wakaflocka17/bert-imdb-finetuned",
    "bart_base": "wakaflocka17/bart-imdb-finetuned",
    "gpt_neo_2_7b": "wakaflocka17/gptneo-imdb-finetuned"
}

os.makedirs("models", exist_ok=True)

for local_name, hf_repo in models.items():
    save_dir = os.path.join("models", local_name)
    print(f"⬇️ Scaricando: {hf_repo} → {save_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForSequenceClassification.from_pretrained(hf_repo)

    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

print("✅ Tutti i modelli sono stati salvati nella cartella 'models/'")