import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

def load_imdb_dataset():
    """
    Carica il dataset IMDb tramite la libreria `datasets`.
    Ritorna un DatasetDict con train, test e (opzionale) unsupervised.
    """
    logging.info("Caricamento dataset IMDb da Hugging Face...")
    dataset = load_dataset("stanfordnlp/imdb")
    # If exists, remove the unsupervised split
    if "unsupervised" in dataset:
        del dataset["unsupervised"]
    return dataset


def create_splits(dataset, val_ratio=0.2, seed=42):
    """
    Suddivide il dataset di training in train + validation (es. 80-20).
    Ritorna: train_dataset, val_dataset, test_dataset
    """
    logging.info("Creazione split train/val/test...")
    dataset_split = dataset["train"].train_test_split(test_size=val_ratio, seed=seed)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]
    test_dataset = dataset["test"]
    logging.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

def preprocess_text(text):
    """
    Esempio di funzione di preprocessing, se vuoi pulire o normalizzare il testo.
    Qui lasciata vuota o con una semplice lower().
    """
    return text.lower()