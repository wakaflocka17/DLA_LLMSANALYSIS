import numpy as np
import logging
import os
from transformers import BartForSequenceClassification, BartTokenizer, Trainer, TrainingArguments
from src.utils import TqdmLoggingCallback
from src.data_preprocessing import load_imdb_dataset

# Import dal modulo sklearn per le metriche
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BartBaseIMDB:
    def __init__(self, repo_finetuned: str, repo_pretrained: str, pretrained_model_name: str = "facebook/bart-base", **kwargs):
        """
        Inizializza il modello BART per la classificazione sul dataset IMDB.
        
        Parametri:
          - repo_finetuned (str): Repository in cui salvare il modello fine-tunato.
          - repo_pretrained (str): Repository (locale) in cui salvare o recuperare il modello pre-addestrato (se desiderato).
          - pretrained_model_name (str): Nome del modello pre-addestrato da caricare.
          - kwargs: Parametri aggiuntivi opzionali.
        """
        self.repo_finetuned = repo_finetuned
        self.repo_pretrained = repo_pretrained
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_model_name)
        # Imposta il modello per la classificazione binaria (num_labels=2)
        self.model = BartForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
        self.train_dataset = None
        self.eval_dataset = None

    def prepare_datasets(self, dataset_name: str = "imdb", split_train: str = "train", split_test: str = "test", max_samples: int = None):
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
        Calcola e logga accuracy, precision, recall, F1 e classification_report.
        """
        logits, labels = eval_pred

        # Conversione diretta a NumPy, con eventuale squeeze se arrivano in 3D
        logits = np.array(logits, dtype=np.float32)
        if logits.ndim == 3:
            # Da shape (batch, num_labels, 1) a (batch, num_labels)
            logits = np.squeeze(logits, axis=-1)

        predictions = np.argmax(logits, axis=-1)
        
        accuracy = np.mean(predictions == labels)
        precision = precision_score(labels, predictions, average='binary')
        recall = recall_score(labels, predictions, average='binary')
        f1 = f1_score(labels, predictions, average='binary')

        # Calcolo del classification report
        report = classification_report(
            labels,
            predictions,
            target_names=["negativo", "positivo"],
            digits=4
        )

        # Logga il classification report
        logger.info("\n" + report)

        # Ritorna le metriche principali
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def train(self, output_dir: str = "./results", num_train_epochs: int = 3, per_device_train_batch_size: int = 8, **kwargs):
        if self.train_dataset is None or self.eval_dataset is None:
            self.prepare_datasets()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            evaluation_strategy="epoch",   # Valutazione a fine epoca
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
        # Salva il modello fine-tunato nella cartella repo_finetuned
        trainer.save_model(self.repo_finetuned)
        if trainer.state.log_history:
            final_log = trainer.state.log_history[-1]
            logger.info(f"Training completato con metriche finali: {final_log}")
        else:
            logger.info("Training completato senza log di metriche finali.")

    def evaluate(self, per_device_eval_batch_size: int = 8, **kwargs):
        """
        Valuta il modello fine-tunato:
          - Carica il modello dai file salvati in repo_finetuned.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        if os.path.exists(self.repo_finetuned):
            logger.info(f"Carico il modello fine-tunato da {self.repo_finetuned}")
            self.model = BartForSequenceClassification.from_pretrained(self.repo_finetuned)

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
        Valuta il modello pre-addestrato:
          - Se si desidera, si può anche salvare il modello pre-addestrato in repo_pretrained,
            altrimenti viene caricato direttamente da pretrained_model_name.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        eval_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=per_device_eval_batch_size,
            disable_tqdm=False,
            **kwargs
        )
        
        logger.info("Valutazione sul modello pre-addestrato...")
        # Se hai configurato repo_pretrained e la cartella esiste, carica da lì;
        # altrimenti, carica direttamente dal pretrained_model_name.
        if os.path.exists(self.repo_pretrained):
            pretrained_model = BartForSequenceClassification.from_pretrained(self.repo_pretrained, num_labels=2)
            logger.info(f"Carico il modello pre-addestrato da {self.repo_pretrained}")
        else:
            pretrained_model = BartForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=2)
            logger.info(f"Carico il modello pre-addestrato da {self.pretrained_model_name}")

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