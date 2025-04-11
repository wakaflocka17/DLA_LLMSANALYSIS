import logging
import os
import tarfile
import urllib.request
from datasets import Dataset, DatasetDict
import pandas as pd
import shutil

logging.basicConfig(level=logging.INFO)

def load_imdb_dataset():
    """
    Carica il dataset IMDb scaricandolo direttamente da Stanford.
    Ritorna un DatasetDict con train e test splits.
    """
    logging.info("Caricamento dataset IMDb direttamente da Stanford...")
    
    # URL and local paths
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    data_dir = "/Users/wakaflocka/Desktop/UNIVERSITÀ/QUINTO ANNO/SECONDO SEMESTRE/Deep Learning and Applications/Sentiment_Analysis_Project/data"
    tar_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    extract_path = os.path.join(data_dir, "aclImdb")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset if not already downloaded
    if not os.path.exists(tar_path):
        logging.info(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        logging.info("Download completed.")
    else:
        logging.info("Dataset archive already exists, skipping download.")
    
    # Extract the dataset if not already extracted
    if not os.path.exists(extract_path):
        logging.info("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        logging.info("Extraction completed.")
    else:
        logging.info("Dataset already extracted, skipping extraction.")
    
    # Convert the dataset to a format compatible with the rest of the code
    train_data = []
    test_data = []
    
    # Process training data
    logging.info("Processing training data...")
    for sentiment in ['pos', 'neg']:
        label = 1 if sentiment == 'pos' else 0
        dir_path = os.path.join(extract_path, 'train', sentiment)
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    train_data.append({"text": text, "label": label})
    
    # Process test data
    logging.info("Processing test data...")
    for sentiment in ['pos', 'neg']:
        label = 1 if sentiment == 'pos' else 0
        dir_path = os.path.join(extract_path, 'test', sentiment)
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    test_data.append({"text": text, "label": label})
    
    # Convert to pandas DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create a DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    logging.info(f"Dataset caricato con successo: {dataset}")
    logging.info(f"Split disponibili: {dataset.keys()}")
    logging.info(f"Numero di esempi - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    
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