from src.data_preprocessing import load_imdb_dataset, create_splits
import numpy as np
import logging
import os
from transformers import Trainer, TrainingArguments
from sklearn.metrics import classification_report
from scipy.stats import mode

from model_factory import get_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EnsembleMajorityVoting:
    def __init__(self, repo: str, member_names: list = None, **kwargs):
        self.repo = repo
        if member_names is None:
            member_names = ["bert-base-uncased-imdb", "bart-base-imdb", "gpt-neo-2.7b-imdb"]
        self.members = [get_model(name, **kwargs) for name in member_names]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_datasets(self, max_samples: int = None):
        full_dataset = load_imdb_dataset()
        train_dataset, val_dataset, test_dataset = create_splits(full_dataset)

        if max_samples:
            train_dataset = train_dataset.shuffle(seed=42).select(range(max_samples))
            val_dataset = val_dataset.shuffle(seed=42).select(range(max_samples))
            test_dataset = test_dataset.shuffle(seed=42).select(range(max_samples))

        for member in self.members:
            member.prepare_datasets(max_samples=max_samples)

        # Usiamo il dataset del primo membro come riferimento
        self.train_dataset = self.members[0].train_dataset
        self.val_dataset = self.members[0].val_dataset
        self.test_dataset = self.members[0].test_dataset

    def train(self, output_dir: str = "./results", per_device_train_batch_size: int = 8, **kwargs):
        logger.info("Inizio training per l'ensemble. Allenamento dei membri:")
        for member in self.members:
            logger.info(f"--> Training del membro: {member.__class__.__name__}")
            member.train(output_dir=output_dir, per_device_train_batch_size=per_device_train_batch_size, **kwargs)
        logger.info("Training dell'ensemble completato.")

    def predict(self, per_device_eval_batch_size: int = 8, **kwargs):
        if self.test_dataset is None:
            self.prepare_datasets()

        predictions_dict = {}
        member_preds = {}

        # Otteniamo le predizioni per ogni membro
        for member in self.members:
            eval_args = TrainingArguments(
                output_dir="./results",
                per_device_eval_batch_size=per_device_eval_batch_size,
                disable_tqdm=True,
                **kwargs
            )
            trainer = Trainer(
                model=member.model,
                args=eval_args,
                eval_dataset=self.test_dataset,
                tokenizer=member.tokenizer
            )
            preds = trainer.predict(self.test_dataset).predictions
            preds_labels = np.argmax(preds, axis=-1)
            member_preds[member.repo_finetuned.split("/")[-1]] = preds_labels

        # Costruisci il dizionario per ogni sample
        num_samples = len(self.test_dataset)
        true_labels = self.test_dataset["label"]

        ensemble_predictions = []
        for i in range(num_samples):
            sample_pred = {"true_label": true_labels[i]}
            votes = []
            for model_name, preds in member_preds.items():
                sample_pred[model_name] = preds[i]
                votes.append(preds[i])
            voted, _ = mode(votes)
            sample_pred["voted"] = int(voted[0])
            ensemble_predictions.append(sample_pred)

        return ensemble_predictions

    def evaluate(self, per_device_eval_batch_size: int = 8, output_json_path: str = None, **kwargs):
        if self.test_dataset is None:
            self.prepare_datasets()

        detailed_preds = self.predict(per_device_eval_batch_size, **kwargs)
        ensemble_preds = [sample["voted"] for sample in detailed_preds]
        true_labels = [sample["true_label"] for sample in detailed_preds]

        accuracy = np.mean(np.array(ensemble_preds) == np.array(true_labels))
        logger.info(f"Ensemble evaluation accuracy: {accuracy:.4f}")

        report = classification_report(
            true_labels,
            ensemble_preds,
            target_names=["negativo", "positivo"],
            digits=4
        )
        logger.info("\n" + report)

        results = {
            "accuracy": accuracy,
            "detailed_predictions": detailed_preds
        }

        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved ensemble evaluation results to {output_json_path}")

        return results