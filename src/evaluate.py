import logging
import numpy as np
import evaluate
import os
import torch
from src.utils import get_model_type
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from huggingface_hub import HfFolder

logging.basicConfig(level=logging.INFO)

def evaluate_model(model_path, dataset, is_pretrained=False, cache_dir=None):
    """
    Valutazione adattata per tipologia di modello (encoder-only, encoder-decoder, decoder-only).
    """
    logging.info(f"Caricamento tokenizer e modello da: {model_path}")

    try:
        hf_token = HfFolder.get_token()
        if hf_token:
            logging.info("🔑 Token Hugging Face trovato.")

        load_options = {"num_labels": 2}
        if cache_dir:
            load_options["cache_dir"] = cache_dir

        if os.path.exists(model_path):
            load_options["local_files_only"] = True

        tokenizer = AutoTokenizer.from_pretrained(model_path, **load_options)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **load_options)

    except Exception as e:
        logging.error(f"Errore nel caricamento del modello {model_path}: {e}")
        raise RuntimeError(f"Impossibile caricare il modello {model_path}: {e}")

    model_type = get_model_type(model_path)
    logging.info(f"Tipologia del modello identificata: {model_type}")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    preds, refs = [], []

    for idx, example in enumerate(dataset):
        input_ids = example["input_ids"].unsqueeze(0)
        attention_mask = example["attention_mask"].unsqueeze(0)
        labels = example["labels"].item()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu()

            if model_type in ["encoder-only", "encoder-decoder"]:
                pred = logits.argmax(dim=-1).item()
            elif model_type == "decoder-only":
                pred = logits[:, -1, :].argmax(dim=-1).item()
            else:
                raise ValueError("Tipo modello non supportato")

        preds.append(pred)
        refs.append(labels)

        if idx % 500 == 0 and idx != 0:
            logging.info(f"🧮 Valutati {idx}/{len(dataset)} esempi")

    accuracy = accuracy_metric.compute(predictions=preds, references=refs)
    f1 = f1_metric.compute(predictions=preds, references=refs, average="binary")

    logging.info("\n" + classification_report(refs, preds, target_names=["negativo", "positivo"]))

    logging.info(f"✅ Accuracy: {accuracy['accuracy']:.4f} | F1: {f1['f1']:.4f}")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}