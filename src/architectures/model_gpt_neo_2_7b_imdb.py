import numpy as np
import logging
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
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
        # Impostiamo il token di padding (usiamo eos_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Carica il modello per la sequence classification; num_labels=2 per classificazione binaria
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
        Calcola le metriche di valutazione (ad es. accuratezza).
        
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
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_steps=100,
            disable_tqdm=True,  # Puoi modificare questo parametro se desideri la barra di avanzamento custom
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
        Esegue l'evaluation sul modello fine-tunato (carica i pesi dalla directory di salvataggio)
        con logging dettagliato e progress bar.
        
        Parametri:
          - per_device_eval_batch_size (int): Batch size per la valutazione.
          - kwargs: Parametri aggiuntivi per TrainingArguments.
        
        Ritorna:
          - Un dizionario contenente le metriche di evaluation.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        # Se la directory fine-tunata esiste, carica i pesi aggiornati
        if os.path.exists(self.repo):
            from transformers import AutoModelForSequenceClassification
            logger.info(f"Carico il modello fine-tunato da {self.repo}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.repo)

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
        logger.info("Inizio valutazione (fine-tunato)...")
        results = trainer.evaluate()
        logger.info(f"Valutazione completata con risultati: {results}")
        return results

    def evaluate_pretrained(self, per_device_eval_batch_size: int = 8, **kwargs):
        """
        Esegue l'evaluation sul modello pre-addestrato, senza caricare i pesi fine-tunati.
        
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

        # Ricarica il modello pre-addestrato, in modo da non usare i pesi aggiornati durante il training
        from transformers import AutoModelForSequenceClassification
        logger.info("Valutazione sul modello pre-addestrato...")
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=2)
        
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