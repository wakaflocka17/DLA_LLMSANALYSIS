import logging
import numpy as np
from datasets import Dataset, load_dataset
from src.utils import get_model_type
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
# Assicurati di avere anche importato la funzione create_splits se non è presente in questo file
from src.data_preprocessing import create_splits

logging.basicConfig(level=logging.INFO)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

def train_model(
    model_name_or_path: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str = "model_output",
    num_labels: int = 2,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5
):
    """
    Fine-tuning adattato per tipologia di modello (encoder-only, encoder-decoder, decoder-only).
    """

    model_type = get_model_type(model_name_or_path)
    logging.info(f"Tipologia del modello identificata: {model_type}")

    # Caricamento tokenizer (lo stesso per ogni tipo di modello)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Caricamento modello con problem_type per gestione coerente dei logits
    if model_type == "encoder-decoder":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_total_limit=3,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    logging.info("Inizio training...")
    trainer.train()

    logging.info("Training completato. Salvataggio del modello...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Modello e tokenizer salvati in: {output_dir}")

    return trainer

def main(model_name: str, output_dir: str):
    logging.info("Caricamento dataset IMDb...")
    dataset = load_dataset("stanfordnlp/imdb")
    train_data, val_data, test_data = create_splits(dataset)

    logging.info(f"Dimensioni dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    trainer = train_model(
        model_name_or_path=model_name,
        train_dataset=train_data,
        val_dataset=val_data,
        output_dir=output_dir,
        num_labels=2,
        epochs=2,
        batch_size=8,
        lr=2e-5
    )

if __name__ == "__main__":
    main("bert-base-uncased", "./models/bert-base-uncased-imdb")