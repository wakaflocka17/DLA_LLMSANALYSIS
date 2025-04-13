import logging
import numpy as np
import evaluate
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from huggingface_hub import HfFolder

logging.basicConfig(level=logging.INFO)

def evaluate_model(model_path, dataset, is_pretrained=False, cache_dir=None):
    """
    Carica e valuta un modello Hugging Face (locale o remoto) su un dataset.
    Ritorna un dizionario con accuracy, f1 e stampa il classification report.
    
    Args:
        model_path: Percorso al modello (locale o HF Hub ID)
        dataset: Dataset da valutare
        is_pretrained: Se True, usa il modello pre-addestrato senza fine-tuning
        cache_dir: Directory di cache opzionale per i modelli scaricati
    """
    logging.info(f"Caricamento tokenizer e modello da: {model_path}")
    
    try:
        # Ottieni token HF se disponibile
        hf_token = HfFolder.get_token()
        
        # Opzioni di caricamento
        load_options = {
            "num_labels": 2,
        }
        
        # Aggiungi opzioni per cache e token se disponibili
        if cache_dir:
            load_options["cache_dir"] = cache_dir
            logging.info(f"Utilizzo directory cache: {cache_dir}")
            
        if hf_token:
            load_options["use_auth_token"] = hf_token
            logging.info("Token di autenticazione HF trovato e utilizzato")
        
        # Caricamento modello e tokenizer
        if os.path.exists(model_path):
            logging.info(f"Modello trovato localmente in: {model_path}")
            load_options["local_files_only"] = True
            tokenizer = AutoTokenizer.from_pretrained(model_path, **load_options)
            model = AutoModelForSequenceClassification.from_pretrained(model_path, **load_options)
        else:
            logging.info(f"Modello non trovato localmente. Download da Hugging Face: {model_path}")
            # Prova prima senza token
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, **load_options)
                model = AutoModelForSequenceClassification.from_pretrained(model_path, **load_options)
            except Exception as e:
                logging.warning(f"Errore nel download senza token: {e}")
                logging.info("Tentativo di download senza opzioni aggiuntive...")
                # Fallback: prova con opzioni minime
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        
        if is_pretrained:
            logging.info("⚠️ Modalità zero-shot attiva (pre-trained model senza fine-tuning)")

    except Exception as e:
        logging.error(f"❌ Errore nel caricamento del modello {model_path}: {e}")
        logging.error("Suggerimento: Verifica la connessione internet o scarica manualmente i file del modello")
        raise RuntimeError(f"Impossibile caricare il modello {model_path}: {e}")

    # Tokenizzazione del dataset
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Metriche
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # Valutazione
    preds, refs = [], []

    for example in dataset:
        input_ids = example["input_ids"].unsqueeze(0)
        attention_mask = example["attention_mask"].unsqueeze(0)
        labels = example["labels"].item()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
            pred = int(np.argmax(logits, axis=-1)[0])
            preds.append(pred)
            refs.append(labels)

    # Debug check
    assert all(isinstance(p, int) for p in preds), "Errore: preds contiene valori non interi"
    assert all(isinstance(r, int) for r in refs), "Errore: refs contiene valori non interi"

    # Calcolo metriche
    accuracy = accuracy_metric.compute(predictions=preds, references=refs)
    f1 = f1_metric.compute(predictions=preds, references=refs, average="binary")

    # 📊 Classification Report
    logging.info("\n" + classification_report(refs, preds, target_names=["negativo", "positivo"]))

    logging.info(f"✅ Accuracy: {accuracy['accuracy']:.4f} | F1: {f1['f1']:.4f}")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}