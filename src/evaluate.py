import logging
import numpy as np
import evaluate
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)

def evaluate_model(model_path, dataset, is_pretrained=False):
    """
    Carica e valuta un modello Hugging Face (locale o remoto) su un dataset.
    Ritorna un dizionario con accuracy, f1 e stampa il classification report.
    """
    logging.info(f"Caricamento tokenizer e modello da: {model_path}")
    
    try:
        # Caricamento modello e tokenizer
        if os.path.exists(model_path):
            logging.info(f"Modello trovato localmente in: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=2, local_files_only=True
            )
        else:
            logging.info(f"Modello non trovato localmente. Download da Hugging Face: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=2
            )
        
        if is_pretrained:
            logging.info("⚠️ Modalità zero-shot attiva (pre-trained model senza fine-tuning)")

    except Exception as e:
        logging.error(f"❌ Errore nel caricamento del modello {model_path}: {e}")
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