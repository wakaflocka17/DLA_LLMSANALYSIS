import logging
import numpy as np
from datasets import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)

def evaluate_model(model_path, dataset, is_pretrained=False):
    """
    This function loads the model from 'model_path' and evaluates it on 'dataset'.
    If is_pretrained=True, evaluates the pre-trained model in zero-shot setting.
    Returns a dictionary with metrics (accuracy, f1).
    """
    logging.info(f"Caricamento tokenizer e modello da: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # For pre-trained models, we need to specify the number of labels
    if is_pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        logging.info(f"Modello pre-addestrato caricato in modalità zero-shot")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

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