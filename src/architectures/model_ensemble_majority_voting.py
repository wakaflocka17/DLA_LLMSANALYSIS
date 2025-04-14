import numpy as np
import logging
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from scipy.stats import mode

# Importa la funzione factory per ottenere le istanze dei singoli modelli
from model_factory import get_model

# Configuriamo il logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EnsembleMajorityVoting:
    def __init__(self, repo: str, member_names: list = None, **kwargs):
        """
        Inizializza l'ensemble per la classificazione sul dataset IMDB tramite majority voting.

        Parametri:
          - repo (str): Repository Hugging Face dove salvare (eventualmente) l'ensemble.
          - member_names (list): Lista dei nomi dei modelli da includere.
            Default: ["bert-base-uncased-imdb", "bart-base-imdb", "gpt-neo-2.7b-imdb"]
          - kwargs: Parametri aggiuntivi eventualmente da passare alla factory dei modelli.
        """
        self.repo = repo
        if member_names is None:
            member_names = ["bert-base-uncased-imdb", "bart-base-imdb", "gpt-neo-2.7b-imdb"]
        # Istanzia ogni modello membro tramite la factory
        self.members = [get_model(name, **kwargs) for name in member_names]
        self.train_dataset = None
        self.eval_dataset = None

    def prepare_datasets(self, dataset_name: str = "imdb", split_train: str = "train", split_test: str = "test", max_samples: int = None):
        """
        Prepara i dataset di training ed evaluation delegando la preparazione a ciascun membro.

        Parametri:
          - dataset_name (str): Nome del dataset (default "imdb").
          - split_train (str): Nome dello split per il training.
          - split_test (str): Nome dello split per l'evaluation.
          - max_samples (int): Opzionale, limita il numero di campioni (utile per debug).
        """
        # Carichiamo il dataset una volta, ma ne deleghiamo la tokenizzazione a ogni modello
        dataset = load_dataset(dataset_name)
        if max_samples:
            train_dataset = dataset[split_train].shuffle(seed=42).select(range(max_samples))
            eval_dataset = dataset[split_test].shuffle(seed=42).select(range(max_samples))
        else:
            train_dataset = dataset[split_train]
            eval_dataset = dataset[split_test]

        # Per ciascun membro usiamo il loro metodo di preparazione
        for member in self.members:
            # Ogni membro prepara i propri dataset (tokenizzazione, etc.)
            member.prepare_datasets(dataset_name, split_train, split_test, max_samples)
        # Assumiamo che tutti i membri abbiano lo stesso formato e usiamo il dataset del primo come riferimento
        self.train_dataset = self.members[0].train_dataset
        self.eval_dataset = self.members[0].eval_dataset

    def train(self, output_dir: str = "./results", **kwargs):
        """
        Allena ciascun modello membro dell'ensemble.

        Parametri:
          - output_dir (str): Directory di output per salvare i modelli.
          - kwargs: Parametri aggiuntivi da passare al metodo train() dei singoli modelli.
        """
        logger.info("Inizio training per l'ensemble. Allenamento dei membri:")
        for member in self.members:
            logger.info(f"--> Training del membro: {member.__class__.__name__}")
            member.train(output_dir=output_dir, **kwargs)
        logger.info("Training dell'ensemble completato.")

    def predict(self, per_device_eval_batch_size: int = 8, **kwargs):
        """
        Esegue la predizione sui dati di evaluation per ciascun membro e aggrega le predizioni con majority voting.

        Parametri:
          - per_device_eval_batch_size (int): Batch size per la predizione.
          - kwargs: Parametri aggiuntivi da passare a TrainingArguments.
        Ritorna:
          - Un array NumPy con le predizioni ensemble.
        """
        # Assicurarsi che il dataset di evaluation sia stato preparato
        if self.eval_dataset is None:
            self.prepare_datasets()

        predictions_list = []
        for member in self.members:
            # Configuriamo TrainingArguments per la fase di predizione
            eval_args = TrainingArguments(
                output_dir="./results",
                per_device_eval_batch_size=per_device_eval_batch_size,
                disable_tqdm=True,
                **kwargs
            )
            trainer = Trainer(
                model=member.model,
                args=eval_args,
                eval_dataset=self.eval_dataset,
                tokenizer=member.tokenizer
            )
            preds = trainer.predict(self.eval_dataset).predictions
            preds_labels = np.argmax(preds, axis=-1)
            predictions_list.append(preds_labels)

        # Aggrega le predizioni: per ogni esempio si sceglie il voto di maggioranza
        stacked = np.stack(predictions_list, axis=0)  # forma: (num_members, num_samples)
        ensemble_preds, _ = mode(stacked, axis=0)
        ensemble_preds = ensemble_preds.flatten()
        return ensemble_preds

    def evaluate(self, per_device_eval_batch_size: int = 8, **kwargs):
        """
        Valuta l'ensemble effettuando majority voting sulle predizioni dei membri.

        Parametri:
          - per_device_eval_batch_size (int): Batch size per la valutazione.
          - kwargs: Parametri aggiuntivi da passare al metodo predict().
        Ritorna:
          - Un dizionario con la metrica 'accuracy'.
        """
        if self.eval_dataset is None:
            self.prepare_datasets()

        ensemble_preds = self.predict(per_device_eval_batch_size, **kwargs)
        # Si assume che la colonna "label" del dataset contenga le etichette vere
        true_labels = self.eval_dataset["label"]
        accuracy = np.mean(ensemble_preds == true_labels)
        logger.info(f"Ensemble evaluation accuracy: {accuracy}")
        return {"accuracy": accuracy}