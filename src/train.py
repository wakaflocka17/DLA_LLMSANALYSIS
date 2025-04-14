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
    # If logits is a tuple, we extract the first element
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

def prepare_gpt_model_for_batching(tokenizer, model):
    """
    Prepara il modello GPT per gestire batch di dimensioni > 1
    configurando correttamente il pad token e ridimensionando gli embedding.
    """
    # Verifica se il tokenizer ha un pad token
    if tokenizer.pad_token is None:
        logging.info("Pad token non trovato nel tokenizer. Impostazione pad_token = eos_token")
        
        # Imposta il pad token uguale all'eos token
        tokenizer.pad_token = tokenizer.eos_token
        
        # Assicurati che il pad_token_id sia impostato correttamente
        model.config.pad_token_id = tokenizer.eos_token_id
        
        # Ridimensiona gli embedding del modello per includere il nuovo token
        model.resize_token_embeddings(len(tokenizer))
        
        logging.info(f"Pad token impostato a: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        logging.info(f"Dimensione vocabolario: {len(tokenizer)}")
        
        # Verifica che il pad token sia stato impostato correttamente
        assert tokenizer.pad_token is not None, "Pad token non impostato correttamente"
        assert model.config.pad_token_id is not None, "Pad token ID non impostato nel config del modello"
        
        # Verifica che il pad token ID corrisponda all'eos token ID
        assert tokenizer.pad_token_id == tokenizer.eos_token_id, "Pad token ID non corrisponde all'EOS token ID"
    
    return tokenizer, model

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
    
    # Prepara il modello GPT per il batching se necessario
    if "gpt" in model_name_or_path.lower() or "neo" in model_name_or_path.lower():
        logging.info(f"Rilevato modello GPT/Neo: {model_name_or_path}")
        tokenizer, model = prepare_gpt_model_for_batching(tokenizer, model)
        
        # Adatta batch size se necessario per modelli molto grandi
        if "2.7b" in model_name_or_path.lower() and batch_size > 1:
            logging.warning(f"Riduzione batch size da {batch_size} a 1 per GPT-Neo 2.7B")
            batch_size = 1
    
    # Funzione di tokenizzazione comune
    def tokenize_fn(batch):
        # Assicurati che il padding sia applicato correttamente
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"  # Assicura che i tensori siano restituiti in formato PyTorch
        )
        
    # Applica la tokenizzazione
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)

    # Rinomina le colonne
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    # Imposta il formato
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Configura gli argomenti di training
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
        disable_tqdm=False,
        # Aggiungi questi parametri per gestire meglio il padding
        dataloader_drop_last=False,
        remove_unused_columns=False
    )

    # Crea il trainer
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