import numpy as np
import logging
import os
from transformers import BartForSequenceClassification, BartTokenizer, Trainer, TrainingArguments
from src.utils import TqdmLoggingCallback
from src.data_preprocessing import load_imdb_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BartBaseIMDB:
    def __init__(self, repo: str, pretrained_model_name: str = "facebook/bart-base", **kwargs):
        """
        Inizializza il modello BART per la classificazione sul dataset IMDB.
        
        Parametri:
          - repo (str): Repository Hugging Face per il salvataggio del modello fine-tunato.
          - pretrained_model_name (str): Nome del modello pre-addestrato da caricare.
          - kwargs: Parametri aggiuntivi opzionali.
        """
        self.repo = repo
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_model_name)
        # num_labels=2 per classificazione binaria (IMDB)
        self.model = BartForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
        self.train_dataset = None
        self.eval_dataset = None

    def prepare_datasets(self, dataset_name: str = "imdb", split_train: str = "train", split_test: str = "test", max_samples: int = None):
        """
        Carica e prepara i dataset per training ed evaluation.
        
        Parametri:
          - dataset_name (str): Nome del dataset (default "imdb").
          - split_train (str): Split per l'addestramento.
          - split_test (str): Split per la valutazione.
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
        Calcola le metriche di valutazione (ad es. accuratezza).
        
        Parametri:
          - eval_pred: Tuple (logits, labels) fornito dal Trainer.
        Ritorna:
          - Dizionario con le metriche.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {"accuracy": accuracy}

    def train(self, output_dir: str = "./results", num_train_epochs: int = 3, per_device_train_batch_size: int = 8, **kwargs):
        """
        Avvia il training del modello con logging dettagliato e progress bar.
        
        Parametri:
          - output_dir (str): Directory di output per i risultati.
          - num_train_epochs (int): Numero di epoche.
          - per_device_train_batch_size (int): Batch size per dispositivo durante il training.
          - kwargs: Parametri aggiuntivi per TrainingArguments.
        """
        if self.train_dataset is None or self.eval_dataset is None:
            self.prepare_datasets()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,  # Specifico per il training
            evaluation_strategy="epoch",
            logging_steps=100,
            save_strategy="epoch",
            load_best_model_at_end=True,
            disable_tqdm=True,
            **kwargs
        )

        tqdm_callback = TqdmLoggingCallback(update_every=100)

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
        if trainer.state.log_history:
            final_log = trainer.state.log_history[-1]
            logger.info(f"Training completato con metriche finali: {final_log}")
        else:
            logger.info("Training completato senza log di metriche finali.")

    def evaluate(self, per_device_eval_batch_size: int = 8, **kwargs):
        """
        Esegue l'evaluation sul modello fine-tunato: se esiste la directory dei pesi, carica i pesi aggiornati
        e usa il modello fine-tunato per eseguire l'evaluation.
        
        Parametri:
          - per_device_eval_batch_size (int): Batch size per dispositivo durante l'evaluation.
          - kwargs: Parametri aggiuntivi per TrainingArguments.
        
        Ritorna:
          - Dizionario con le metriche di evaluation.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        if os.path.exists(self.repo):
            from transformers import BartForSequenceClassification
            logger.info(f"Carico il modello fine-tunato da {self.repo}")
            self.model = BartForSequenceClassification.from_pretrained(self.repo)

        eval_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=per_device_eval_batch_size,  # Specifico per l'evaluation
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
        logger.info("Inizio valutazione (fine-tunato)...")
        results = trainer.evaluate()
        logger.info(f"Valutazione completata con risultati: {results}")
        return results

    def evaluate_pretrained(self, per_device_eval_batch_size: int = 8, **kwargs):
        """
        Esegue l'evaluation sul modello pre-addestrato, senza caricare i pesi fine-tunati.
        
        Parametri:
          - per_device_eval_batch_size (int): Batch size per dispositivo durante l'evaluation.
          - kwargs: Parametri aggiuntivi per TrainingArguments.
        
        Ritorna:
          - Dizionario con le metriche di evaluation.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        eval_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=per_device_eval_batch_size,
            disable_tqdm=False,
            **kwargs
        )
        from transformers import BartForSequenceClassification
        logger.info("Valutazione sul modello pre-addestrato...")
        pretrained_model = BartForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=2)
        
        trainer = Trainer(
            model=pretrained_model,
            args=eval_args,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        logger.info("Inizio valutazione (pre-addestrato)...")
        results = trainer.evaluate()
        logger.info(f"Valutazione completata con risultati: {results}")
        return results