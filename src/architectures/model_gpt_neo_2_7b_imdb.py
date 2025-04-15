import numpy as np
import logging
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.utils import TqdmLoggingCallback
from src.data_preprocessing import load_imdb_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GPTNeo27BIMDB:
    def __init__(self, repo_finetuned: str, repo_pretrained: str, pretrained_model_name: str = "EleutherAI/gpt-neo-2.7B", **kwargs):
        """
        Inizializza il modello GPT-Neo 2.7B per la classificazione sul dataset IMDB.
        
        Parametri:
          - repo_finetuned (str): Repository in cui salvare il modello fine-tunato.
          - repo_pretrained (str): Repository (locale) da cui recuperare il modello pre-addestrato (se desiderato).
          - pretrained_model_name (str): Nome del modello pre-addestrato da caricare.
          - kwargs: Parametri aggiuntivi opzionali.
        """
        self.repo_finetuned = repo_finetuned
        self.repo_pretrained = repo_pretrained
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        # Impostiamo il token di padding (usiamo eos_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Carica il modello per la sequence classification (num_labels=2 per classificazione binaria)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
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
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {"accuracy": accuracy}

    def train(self, output_dir: str = "./results", num_train_epochs: int = 3, per_device_train_batch_size: int = 8, **kwargs):
        """
        Esegue il training del modello fine-tunato.
        Al termine, salva i pesi nella directory repo_finetuned.
        """
        if self.train_dataset is None or self.eval_dataset is None:
            self.prepare_datasets()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            evaluation_strategy="epoch",
            logging_steps=100,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
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
        # Salva il modello fine-tunato nella directory repo_finetuned
        trainer.save_model(self.repo_finetuned)
        if trainer.state.log_history:
            final_log = trainer.state.log_history[-1]
            logger.info(f"Training completato con metriche finali: {final_log}")
        else:
            logger.info("Training completato senza log di metriche finali.")

    def evaluate(self, per_device_eval_batch_size: int = 8, **kwargs):
        """
        Valuta il modello fine-tunato: se la directory repo_finetuned esiste, carica i pesi da lì.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        if os.path.exists(self.repo_finetuned):
            from transformers import AutoModelForSequenceClassification
            logger.info(f"Carico il modello fine-tunato da {self.repo_finetuned}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.repo_finetuned)

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
          - Se la directory repo_pretrained esiste, la usa (se vuoi salvare localmente il pre-addestrato),
            altrimenti carica direttamente dal pretrained_model_name.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        eval_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=per_device_eval_batch_size,
            disable_tqdm=False,
            **kwargs
        )
        from transformers import AutoModelForSequenceClassification
        logger.info("Valutazione sul modello pre-addestrato...")
        if os.path.exists(self.repo_pretrained):
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.repo_pretrained, num_labels=2)
            logger.info(f"Carico il modello pre-addestrato da {self.repo_pretrained}")
        else:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=2)
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