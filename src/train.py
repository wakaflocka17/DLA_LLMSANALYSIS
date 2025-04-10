import logging
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset, load_metric

logging.basicConfig(level=logging.INFO)

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    """
    Computes accuracy and F1 score for a given set of predictions and labels.
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

    # Model and tokenizer initialization
    logging.info(f"Caricamento tokenizer e modello da: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels
    )

    # Tokenization function
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # We apply the tokenization to the datasets
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    # We rename the "label" column to "labels"
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    # We set the PyTorch format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # We define the training arguments
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
        logging_steps=50
    )

    # We define the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # We start training
    logging.info("Inizio training...")
    trainer.train()

    # We save the model and tokenizer
    logging.info("Training completato. Salvataggio del modello...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Modello e tokenizer salvati in: {output_dir}")

    return trainer

def main():
    """
    This is the main function in which we load the IMDb dataset, 
    create a split train/val, and then call train_model() for fine-tuning.
    """
    logging.info("Caricamento dataset IMDb...")
    dataset = load_dataset("imdb")

    # We create a split equal to 20% of the size of the split provided for the train.
    dataset_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_data = dataset_split["train"]
    val_data = dataset_split["test"]
    test_data = dataset["test"]  # 25k esempi

    logging.info(f"Dimensioni dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # We define a model name and output directory.
    model_name = "bert-base-uncased"
    output_dir = "bert_imdb"

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
    main()