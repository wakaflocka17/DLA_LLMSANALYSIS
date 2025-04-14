import random
import numpy as np
import torch
from huggingface_hub import create_repo, upload_folder, upload_file
import os
import logging
from tqdm import tqdm
from transformers import TrainerCallback

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
        # Nel caso decoder-only potrebbe essere necessario prendere l'ultimo token
        return logits[:, -1, :].argmax(dim=-1).item()
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
    """
    Callback personalizzato che utilizza tqdm per visualizzare una barra di avanzamento
    e logga i dettagli dei progressi e delle metriche ad ogni step.
    """
    def __init__(self):
        self.tqdm_bar = None
        self.last_logged_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # Calcola il numero totale di step; se non definito, calcola in base al dataset e batch size
        total_steps = state.max_steps if state.max_steps and state.max_steps > 0 else int(
            (len(kwargs.get("train_dataset", [])) / args.per_device_train_batch_size) * args.num_train_epochs
        )
        self.tqdm_bar = tqdm(total=total_steps, desc="Training")
        logger.info(f"Inizio training: {total_steps} step totali.")

    def on_step_end(self, args, state, control, **kwargs):
        if self.tqdm_bar is not None:
            self.tqdm_bar.update(1)
        if (state.global_step - self.last_logged_step) >= 10:
            self.last_logged_step = state.global_step
            if state.log_history:
                last_log = state.log_history[-1]
                logger.info(f"Step {state.global_step} - Log: {last_log}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.tqdm_bar is not None:
            self.tqdm_bar.close()
        logger.info("Training completato.")