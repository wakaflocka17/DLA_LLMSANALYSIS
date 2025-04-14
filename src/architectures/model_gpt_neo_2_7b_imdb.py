import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from src.utils import TqdmLoggingCallback
from src.data_preprocessing import load_imdb_dataset

# Configurazione del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GPTNeo27BIMDB:
    def __init__(self, repo: str, pretrained_model_name: str = "EleutherAI/gpt-neo-2.7B", **kwargs):
        """
        Inizializza il modello GPT-Neo 2.7B per la classificazione sul dataset IMDB.

        Parametri:
          - repo (str): Repository Hugging Face dove salvare il modello fine-tunato.
          - pretrained_model_name (str): Nome del modello pre-addestrato (default "EleutherAI/gpt-neo-2.7B").
          - kwargs: Parametri aggiuntivi opzionali.
        """
        self.repo = repo
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        # Carica il modello per la sequence classification; impostiamo num_labels=2 per la classificazione binaria
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
        self.train_dataset = None
        self.eval_dataset = None

    def prepare_datasets(self, dataset_name: str = "imdb", split_train: str = "train", split_test: str = "test", max_samples: int = None):
        """
        Carica e prepara i dataset per training ed evaluation.

        Parametri:
          - dataset_name (str): Nome del dataset (default "imdb").
          - split_train (str): Nome dello split per il training.
          - split_test (str): Nome dello split per l'evaluation.
          - max_samples (int): Limita il numero di campioni (utile per debug).
        """
        dataset = load_imdb_dataset()
        if max_samples:
            self.train_dataset = dataset[split_train].shuffle(seed=42).select(range(max_samples))
            self.eval_dataset = dataset[split_test].shuffle(seed=42).select(range(max_samples))
        else:
            self.train_dataset = dataset[split_train]
            self.eval_dataset = dataset[split_test]

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )

        self.train_dataset = self.train_dataset.map(tokenize_function, batched=True)
        self.eval_dataset = self.eval_dataset.map(tokenize_function, batched=True)

    def compute_metrics(self, eval_pred):
        """
        Calcola le metriche di valutazione (es. accuratezza).

        Parametri:
          - eval_pred: Tuple (logits, labels) fornito dal Trainer.
        Ritorna:
          - Un dizionario contenente le metriche.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {"accuracy": accuracy}

    def train(self, output_dir: str = "./results", num_train_epochs: int = 3, per_device_train_batch_size: int = 8, **kwargs):
        """
        Avvia il training del modello con logging dettagliato e progress bar.

        Parametri:
          - output_dir (str): Directory per salvare i risultati.
          - num_train_epochs (int): Numero di epoche di training.
          - per_device_train_batch_size (int): Batch size per dispositivo.
          - kwargs: Parametri aggiuntivi per TrainingArguments.
        """
        if self.train_dataset is None or self.eval_dataset is None:
            self.prepare_datasets()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            evaluation_strategy="epoch",
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True,
            disable_tqdm=False,  # Abilita la barra di avanzamento
            **kwargs
        )

        tqdm_callback = TqdmLoggingCallback()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[tqdm_callback]
        )

        trainer.train()
        trainer.save_model(self.repo)
        # Log finale con il report delle metriche (se presente)
        if trainer.state.log_history:
            final_log = trainer.state.log_history[-1]
            logger.info(f"Training completato con metriche finali: {final_log}")
        else:
            logger.info("Training completato senza log di metriche finali.")

    def evaluate(self, per_device_eval_batch_size: int = 8, **kwargs):
        """
        Esegue l'evaluation del modello con logging dettagliato e progress bar.

        Parametri:
          - per_device_eval_batch_size (int): Batch size per la valutazione.
          - kwargs: Parametri aggiuntivi per TrainingArguments.
        Ritorna:
          - Un dizionario contenente le metriche di evaluation.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        eval_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=per_device_eval_batch_size,
            disable_tqdm=False,
            **kwargs
        )

        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        logger.info("Inizio valutazione...")
        results = trainer.evaluate()
        logger.info(f"Valutazione completata con risultati: {results}")
        return results