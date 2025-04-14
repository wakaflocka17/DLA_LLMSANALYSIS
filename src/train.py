import logging
import numpy as np
import evaluate
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from src.model_factory import get_tokenizer_and_model

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
    # Otteniamo tokenizer e modello in modo modulare
    tokenizer, model = get_tokenizer_and_model(model_name_or_path, num_labels)
    
    # Funzione di tokenizzazione comune
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