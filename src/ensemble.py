import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)

def majority_voting_ensemble(model_paths, dataset):
    """
    This function infers from multiple models (specified with the 'model_paths' parameter) 
    and votes for the final class using the majority voting technique.
    """
    logging.info(f"Caricamento {len(model_paths)} modelli per ensemble...")

    # We load the tokenizers/models in a list
    models = []
    tokenizers = []
    for path in model_paths:
        tokenizers.append(AutoTokenizer.from_pretrained(path))
        models.append(AutoModelForSequenceClassification.from_pretrained(path))

    # NB: if the tokenizers differ, you could use the first as standard
    tokenizer = tokenizers[0]

    # Tokenize and format the dataset
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

    ensemble_preds = []
    ensemble_refs = []

    for example in dataset:
        input_ids = example["input_ids"].unsqueeze(0)
        attention_mask = example["attention_mask"].unsqueeze(0)
        label = example["labels"].item()

        # We get the prediction of each model
        votes = []
        for model in models:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().numpy()
            pred = np.argmax(logits, axis=-1)[0]
            votes.append(pred)

        # Majority voting
        final_pred = np.round(np.mean(votes)).astype(int)
        ensemble_preds.append(final_pred)
        ensemble_refs.append(label)

    # Metrics calculation
    from datasets import load_metric
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    acc = accuracy_metric.compute(predictions=ensemble_preds, references=ensemble_refs)
    f1 = f1_metric.compute(predictions=ensemble_preds, references=ensemble_refs, average="binary")

    logging.info(f"Ensemble Accuracy: {acc['accuracy']:.4f}, F1: {f1['f1']:.4f}")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}