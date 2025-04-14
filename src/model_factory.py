import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import get_model_type

def get_tokenizer_and_model(model_name_or_path: str, num_labels: int = 2):
    model_type = get_model_type(model_name_or_path)
    logging.info(f"Tipologia del modello identificata: {model_type}")
    
    # Caricamento del tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    # Per i modelli decoder-only (come GPT-Neo 2.7B) è fondamentale avere un pad_token
    if model_type == "decoder-only" and tokenizer.pad_token is None:
        logging.info("Il tokenizer non ha un pad_token, aggiungo un token di padding uguale a eos_token")
        # Aggiunge il token EOS come pad_token tramite il metodo add_special_tokens
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    # Opzioni di caricamento specifiche (ad es. ignore_mismatched_sizes per gli encoder-decoder)
    load_options = {"num_labels": num_labels}
    if model_type == "encoder-decoder":
        load_options["ignore_mismatched_sizes"] = True
    
    # Caricamento del modello
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **load_options)
    
    # Se abbiamo aggiunto special tokens al tokenizer, aggiorniamo anche le embedding del modello
    if model_type == "decoder-only" and tokenizer.pad_token is not None:
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model