import logging
import numpy as np

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from datasets import Dataset, load_dataset, load_metric

logging.basicConfig(level=logging.INFO)

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    """
    Calcola accuracy e F1 score dati logits e label.
    """
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
    This function make a fine-tuning for a pre-trained model 
    (es: 'EleutherAI/gpt-neo-2.7B', 'bert-base-uncased', 'facebook/bart-base').
    """

    # We initialize model and tokenizer
    logging.info(f"Caricamento tokenizer e modello da: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels
    )

    # Tokenizer function
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # We apply the tokenizer to the datasets
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    # Rename the 'label' column to 'labels'
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    # We set the format of the datasets to PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # We define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        # Save the checkpoints for all epochs
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # Epochs number
        num_train_epochs=epochs,
        # Batch size
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # Learning rate
        learning_rate=lr,
        # We load the best model at the end
        load_best_model_at_end=True,
        # Log directory
        logging_dir=f"{output_dir}/logs",
        # Logging frequency
        logging_steps=50,
        # The checkpoints limit
        save_total_limit=3,
        # We set the tqdm progress bar
        disable_tqdm=False
    )

    # We create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # We start the training
    logging.info("Inizio training...")
    trainer.train()

    # We save the model and tokenizer
    logging.info("Training completato. Salvataggio del modello...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Modello e tokenizer salvati in: {output_dir}")

    return trainer

def main(model_name: str, output_dir: str):
    """
    This main function handles loading the IMDb dataset, 
    creating the split train/val and starting the fine-tuning with train_model() function.
    """
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
    main("bert-base-uncased", "bert_imdb")