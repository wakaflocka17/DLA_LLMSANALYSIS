import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import get_model_type

def get_tokenizer_and_model(model_name_or_path: str, num_labels: int = 2):
    model_type = get_model_type(model_name_or_path)
    logging.info(f"Tipologia del modello identificata: {model_type}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    # Gestione specifica per modelli decoder-only: assicurarsi che esista un pad_token.
    if model_type == "decoder-only" and tokenizer.pad_token is None:
        logging.info("Il tokenizer non ha un pad_token, impostazione pad_token uguale a eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        
    load_options = {"num_labels": num_labels}
    if model_type == "encoder-decoder":
        load_options["ignore_mismatched_sizes"] = True
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **load_options)
    
    return tokenizer, model