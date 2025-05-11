import random
import numpy as np
import torch
from huggingface_hub import create_repo, upload_folder, upload_file
import os
import logging
from tqdm import tqdm
from transformers import TrainerCallback, AutoModelForSequenceClassification, AutoTokenizer # Added AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# If we want to use a different model, we can change the model type here
ENCODER_ONLY_MODELS = ["bert", "roberta", "distilbert"]
ENCODER_DECODER_MODELS = ["bart", "t5", "llama"]
DECODER_ONLY_MODELS = ["gpt", "gpt-neo"]

def get_model_type(model_name):
    model_name = model_name.lower()
    if any(m in model_name for m in ENCODER_ONLY_MODELS):
        return "encoder-only"
    elif any(m in model_name for m in ENCODER_DECODER_MODELS):
        return "encoder-decoder"
    elif any(m in model_name for m in DECODER_ONLY_MODELS):
        return "decoder-only"
    else:
        raise ValueError(f"Unknown model type for {model_name}")

def get_prediction(logits, model_type: str):
    if model_type in ["encoder-only", "encoder-decoder"]:
        return logits.argmax(dim=-1).item()
    elif model_type == "decoder-only":
        # Per AutoModelForSequenceClassification, i logits sono (batch_size, num_classes)
        # anche per i modelli decoder-only, poiché l'head di classificazione gestisce il pooling.
        # La riga originale `logits[:, -1, :].argmax(dim=-1).item()` presumeva
        # che i logits fossero (batch_size, sequence_length, num_classes), il che non è il caso qui.
        return logits.argmax(dim=-1).item() # Riga corretta
    else:
        raise ValueError("Tipo modello non supportato")

def upload_model_to_hf(model_dir: str, repo_id: str, exist_ok=True, private=True):
    """
    Uploads a model directory or file to the Hugging Face Hub.
    """
    logging.info(f"Uploading {model_dir} to Hugging Face as {repo_id}...")

    create_repo(repo_id, exist_ok=exist_ok, private=private)

    if os.path.isfile(model_dir):
        upload_file(
            path_or_fileobj=model_dir,
            path_in_repo=os.path.basename(model_dir),
            repo_id=repo_id,
            repo_type="model"
        )
    else:
        upload_folder(
            folder_path=model_dir,
            path_in_repo="",
            repo_id=repo_id,
            repo_type="model"
        )

    logging.info(f"Upload completato per {repo_id}")

def set_seed(seed=42):
    """
    With this function we can set the PyTorch,
    Numpy and Python random seed.
    :param seed: int, random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TqdmLoggingCallback(TrainerCallback):
    def __init__(self, update_every=10):
        self.pbar = None
        self.last_update_step = 0
        self.update_every = update_every
        self.training_loss = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps if state.max_steps and state.max_steps > 0 else int(
            (len(kwargs.get("train_dataset", [])) / args.per_device_train_batch_size) * args.num_train_epochs
        )
        # Format the progress bar to match the desired output
        self.pbar = tqdm(
            total=total_steps,
            desc="Training",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Loss: {postfix[0]:.4f}]",
            mininterval=1.0,
            postfix=[0.0]  # Initial loss value
        )
        logger.info(f"Inizio training: {total_steps} step totali.")

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar is not None and state.global_step is not None:
            steps_since_update = state.global_step - self.last_update_step
            
            # Update loss value if available
            if state.log_history and len(state.log_history) > 0:
                latest_log = state.log_history[-1]
                if 'loss' in latest_log:
                    self.training_loss = latest_log['loss']
                    self.pbar.postfix[0] = self.training_loss
            
            if steps_since_update >= self.update_every:
                self.pbar.update(steps_since_update)
                self.last_update_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar is not None:
            if state.global_step > self.last_update_step:
                self.pbar.update(state.global_step - self.last_update_step)
            self.pbar.close()
        logger.info("Training completato.")


# New function to load models locally first
def load_local_model(model_config_entry: dict, model_key_for_log: str):
    """
    Loads a model and tokenizer, prioritizing local paths.
    Order: repo_finetuned -> repo_downloaded -> model_name (from HF Hub).
    Uses local_files_only=True for local attempts.

    Args:
        model_config_entry (dict): The configuration dictionary for the specific model
                                   (e.g., MODEL_CONFIGS['bart_base']).
        model_key_for_log (str): The key of the model (e.g., 'bart_base') for logging.

    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails.
    """
    model_name = model_config_entry.get('model_name')
    repo_finetuned = model_config_entry.get('repo_finetuned')
    repo_downloaded = model_config_entry.get('repo_downloaded')

    paths_to_try = []
    if repo_finetuned:
        paths_to_try.append({'path': repo_finetuned, 'source': 'fine-tuned (local)', 'local_only': True})
    if repo_downloaded:
        paths_to_try.append({'path': repo_downloaded, 'source': 'downloaded (local)', 'local_only': True})
    if model_name: # Fallback to Hugging Face Hub
        paths_to_try.append({'path': model_name, 'source': 'Hugging Face Hub', 'local_only': False}) # Try with network if local fails
        paths_to_try.append({'path': model_name, 'source': 'Hugging Face Hub (local cache attempt)', 'local_only': True}) # Also try local_files_only for model_name

    for config in paths_to_try:
        path_to_load = config['path']
        source_info = config['source']
        local_files_flag = config['local_only']
        logger.info(f"Attempting to load {model_key_for_log} from: {path_to_load} (source: {source_info}, local_files_only={local_files_flag})")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(path_to_load, local_files_only=local_files_flag)
            tokenizer = AutoTokenizer.from_pretrained(path_to_load, local_files_only=local_files_flag)
            logger.info(f"Successfully loaded {model_key_for_log} from {path_to_load} ({source_info})")
            return model, tokenizer
        except OSError:
            logger.warning(f"Failed to load {model_key_for_log} from {path_to_load} ({source_info}) with local_files_only={local_files_flag}. It might not exist or not be a valid model directory.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {model_key_for_log} from {path_to_load} ({source_info}): {e}")

    logger.error(f"Failed to load model {model_key_for_log} from all specified paths.")
    return None, None