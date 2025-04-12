import logging
import numpy as np
import evaluate
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfFolder

logging.basicConfig(level=logging.INFO)

def evaluate_model(model_path, dataset, is_pretrained=False):
    """
    This function loads the model from 'model_path' and evaluates it on 'dataset'.
    If is_pretrained=True, evaluates the pre-trained model in zero-shot setting.
    Returns a dictionary with metrics (accuracy, f1).
    """
    logging.info(f"Caricamento tokenizer e modello da: {model_path}")
    
    try:
        # First try to load locally
        if os.path.exists(model_path):
            logging.info(f"Caricamento modello locale da: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            
            if is_pretrained:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=2, local_files_only=True
                )
                logging.info(f"Modello pre-addestrato caricato in modalità zero-shot")
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, local_files_only=True
                )
        else:
            # If not local, try to download from Hugging Face
            logging.info(f"Modello non trovato localmente, tentativo di download da Hugging Face: {model_path}")
            
            # Check if we have a token
            token = HfFolder.get_token()
            if token is None:
                logging.warning("Nessun token Hugging Face trovato. Alcune funzionalità potrebbero essere limitate.")
                
            # Try to download with token if available
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=token)
            
            if is_pretrained:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=2, use_auth_token=token
                )
                logging.info(f"Modello pre-addestrato caricato in modalità zero-shot")
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, use_auth_token=token
                )
    except Exception as e:
        logging.error(f"Errore nel caricamento del modello {model_path}: {e}")
        raise RuntimeError(f"Impossibile caricare il modello {model_path}: {e}")

    # Tokenization dataset
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

    # Loading metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # Inference loop
    preds, refs = [], []
    for example in dataset:
        input_ids = example["input_ids"].unsqueeze(0)
        attention_mask = example["attention_mask"].unsqueeze(0)
        labels = example["labels"].item()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.detach().numpy()
        pred = np.argmax(logits, axis=-1)[0]
        preds.append(pred)
        refs.append(labels)

    accuracy = accuracy_metric.compute(predictions=preds, references=refs)
    f1 = f1_metric.compute(predictions=preds, references=refs, average="binary")

    logging.info(f"Accuracy: {accuracy['accuracy']:.4f}, F1: {f1['f1']:.4f}")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}