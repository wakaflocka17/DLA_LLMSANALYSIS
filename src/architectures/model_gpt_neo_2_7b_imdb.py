import numpy as np
import logging
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.utils import TqdmLoggingCallback
from src.data_preprocessing import load_imdb_dataset, create_splits

# Import aggiuntivi da sklearn per calcolare classification report e metriche
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

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
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        logger.info(f"PAD token id (tokenizer): {self.tokenizer.pad_token_id}")
        logger.info(f"PAD token id (model config): {self.model.config.pad_token_id}")
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.freeze_layers(20)

    def freeze_layers(self, n_freeze=20):
        logger.info(f"Congelo i primi {n_freeze} blocchi di GPT-Neo.")
        for param in self.model.transformer.h[:n_freeze].parameters():
            param.requires_grad = False

    def prepare_datasets(self, max_samples: int = None):
        full_dataset = load_imdb_dataset()
        train_dataset, val_dataset, test_dataset = create_splits(full_dataset)

        if max_samples:
            train_dataset = train_dataset.shuffle(seed=42).select(range(max_samples))
            val_dataset = val_dataset.shuffle(seed=42).select(range(max_samples))
            test_dataset = test_dataset.shuffle(seed=42).select(range(max_samples))

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=64
            )

        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = val_dataset.map(tokenize_function, batched=True)
        self.test_dataset = test_dataset.map(tokenize_function, batched=True)
        
    def compute_metrics(self, eval_pred):
        """
        Versione ottimizzata che calcola solo accuracy durante il training
        per velocizzare la valutazione tra le epoche.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calcola solo accuracy per velocizzare la valutazione durante il training
        accuracy = np.mean(predictions == labels)
        
        # Ritorna solo accuracy per il monitoraggio durante il training
        return {
            "accuracy": accuracy
        }
    
    def evaluate_final(self, model=None):
        """
        Esegue il calcolo completo delle metriche e logga il classification report sul test set.
        """
        if self.test_dataset is None:
            self.prepare_datasets()
            
        eval_model = model if model is not None else self.model

        eval_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=8,
            disable_tqdm=True
        )

        trainer = Trainer(
            model=eval_model,
            args=eval_args,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer
        )

        logger.info("Esecuzione valutazione finale completa sul test set...")
        predictions_output = trainer.predict(self.test_dataset)
        preds = predictions_output.predictions
        labels = predictions_output.label_ids

        final_predictions = np.argmax(preds, axis=-1)

        accuracy = np.mean(final_predictions == labels)
        precision = precision_score(labels, final_predictions, average="binary")
        recall = recall_score(labels, final_predictions, average="binary")
        f1 = f1_score(labels, final_predictions, average="binary")

        logger.info(f"\nMetriche finali complete (test set):\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")

        report = classification_report(
            labels,
            final_predictions,
            target_names=["negativo", "positivo"],
            digits=4
        )
        logger.info("\nClassification Report Test:\n" + report)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report
        }

    def train(self, output_dir: str = "./results", num_train_epochs: int = 3, per_device_train_batch_size: int = 8, **kwargs):
        if self.train_dataset is None or self.val_dataset is None:
            self.prepare_datasets()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            bf16=True,
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
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[tqdm_callback]
        )

        trainer.train()
        trainer.save_model(self.repo_finetuned)

        if trainer.state.log_history:
            final_log = trainer.state.log_history[-1]
            logger.info(f"Training completato con metriche finali: {final_log}")

            training_metrics_path = os.path.join(
                "results", "validation", "finetuned", f"{os.path.basename(self.repo_finetuned)}_metrics.json"
            )
            
            os.makedirs(os.path.dirname(training_metrics_path), exist_ok=True)
            with open(training_metrics_path, "w") as f:
                json.dump(trainer.state.log_history, f, indent=4)
            logger.info(f"Training metrics salvate in {training_metrics_path}")
        else:
            logger.info("Training completato senza log di metriche finali.")

    def evaluate(self, per_device_eval_batch_size: int = 8, output_json_path: str = None, **kwargs):
        if self.test_dataset is None:
            self.prepare_datasets()

        if os.path.exists(self.repo_finetuned):
            logger.info(f"Carico il modello fine-tunato da {self.repo_finetuned}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.repo_finetuned)

        eval_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=per_device_eval_batch_size,
            disable_tqdm=True,
            **kwargs
        )

        logger.info("Inizio valutazione completa (fine-tunato)...")

        results = self.evaluate_final()

        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved fine-tuned evaluation results to {output_json_path}")
        logger.info(f"Valutazione completata con risultati: {results}")
        return results

    def evaluate_pretrained(self, per_device_eval_batch_size: int = 8, output_json_path: str = None, **kwargs):
        if self.test_dataset is None:
            self.prepare_datasets()

        logger.info("Valutazione sul modello pre-addestrato...")
        if os.path.exists(self.repo_pretrained):
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.repo_pretrained, num_labels=2)
            logger.info(f"Carico il modello pre-addestrato da {self.repo_pretrained}")
        else:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=2)
            logger.info(f"Carico il modello pre-addestrato da {self.pretrained_model_name}")

        eval_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=per_device_eval_batch_size,
            disable_tqdm=True,
            **kwargs
        )

        logger.info("Inizio valutazione completa (pre-addestrato)...")

        results = self.evaluate_final()

        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved pretrained evaluation results to {output_json_path}")
        logger.info(f"Valutazione completata con risultati: {results}")
        return results