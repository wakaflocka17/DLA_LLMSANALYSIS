from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys

if len(sys.argv) < 2:
    print("⚠️ Devi specificare il nome del modello: bert_base_uncased, bart_base o gpt_neo_2_7b")
    sys.exit(1)

model_map = {
    "bert_base_uncased": "wakaflocka17/bert-imdb-finetuned",
    "bart_base": "wakaflocka17/bart-imdb-finetuned",
    "gpt_neo_2_7b": "wakaflocka17/gptneo-imdb-finetuned"
}

key = sys.argv[1]
if key not in model_map:
    print(f"❌ Modello '{key}' non valido.")
    sys.exit(1)

hf_repo = model_map[key]
save_dir = os.path.join("models", key)
os.makedirs(save_dir, exist_ok=True)

print(f"⬇️ Scaricando {hf_repo} → {save_dir}")
tokenizer = AutoTokenizer.from_pretrained(hf_repo)
model = AutoModelForSequenceClassification.from_pretrained(hf_repo)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print(f"✅ Modello '{key}' salvato in {save_dir}")