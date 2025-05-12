import random
import numpy as np
import torch
from huggingface_hub import create_repo, upload_folder, upload_file
import os
import logging
from tqdm import tqdm
from transformers import TrainerCallback, AutoModelForSequenceClassification, AutoTokenizer
from optimum.bettertransformer import BetterTransformer # Added
from optimum.onnxruntime import ORTModelForSequenceClassification # Added
import shutil # For cleaning up ONNX export cache if needed

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
def load_local_model(
    model_config_entry: dict, 
    model_key_for_log: str, 
    accelerator, # Added
    use_amp: bool, # Added
    use_bettertransformer: bool, # Added
    use_onnxruntime: bool, # Added
    load_in_8bit_override: bool = False # To allow specific 8-bit loading for models like GPT-Neo
    ):
    """
    Loads a model and tokenizer, prioritizing local paths and applying optimizations.
    If 'repo_finetuned' is specified, it's strictly used and validated.
    Otherwise, falls back to repo_downloaded -> model_name (from HF Hub).
    Uses local_files_only=True for local attempts.

    Args:
        model_config_entry (dict): Config for the model.
        model_key_for_log (str): Model key for logging.
        accelerator (Accelerator): The Accelerator object.
        use_amp (bool): Whether AMP (fp16) is enabled via Accelerator.
        use_bettertransformer (bool): Whether to apply BetterTransformer.
        use_onnxruntime (bool): Whether to attempt loading an ONNX model.
        load_in_8bit_override (bool): Specific flag for 8-bit, e.g. for GPT-Neo.


    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails.
    """
    model_name = model_config_entry.get('model_name')
    repo_finetuned = model_config_entry.get('repo_finetuned')
    repo_downloaded = model_config_entry.get('repo_downloaded')

    paths_to_try = []

    if repo_finetuned:
        logger.info(f"[{model_key_for_log}] 'repo_finetuned' is specified: {repo_finetuned}. Strict checks will be applied.")
        
        # Perform strict checks for the fine-tuned directory as per requirements
        if not os.path.isdir(repo_finetuned):
            raise RuntimeError(f"Fine-tuned model directory not found: {repo_finetuned} for model {model_key_for_log}")

        # Check for essential model and config files
        required_model_files = ["pytorch_model.bin", "config.json", "tokenizer_config.json"]
        for req_file in required_model_files:
            if not os.path.isfile(os.path.join(repo_finetuned, req_file)):
                raise RuntimeError(f"Missing required file '{req_file}' in {repo_finetuned} for model {model_key_for_log}")

        # Check for tokenizer data files (at least one pattern must match)
        has_tokenizer_json = os.path.isfile(os.path.join(repo_finetuned, "tokenizer.json"))
        has_vocab_txt = os.path.isfile(os.path.join(repo_finetuned, "vocab.txt")) # For BERT-like tokenizers
        has_vocab_json = os.path.isfile(os.path.join(repo_finetuned, "vocab.json")) # For BPE-based tokenizers (e.g., GPT-2, BART)
        has_merges_txt = os.path.isfile(os.path.join(repo_finetuned, "merges.txt")) # For BPE-based tokenizers

        if not (has_tokenizer_json or has_vocab_txt or (has_vocab_json and has_merges_txt)):
            missing_files_detail = "tokenizer.json OR vocab.txt OR (vocab.json AND merges.txt)"
            raise RuntimeError(
                f"Missing tokenizer data files ({missing_files_detail}) in {repo_finetuned} for model {model_key_for_log}"
            )
        
        # If all checks passed, this is the only path to try
        paths_to_try.append({'path': repo_finetuned, 'source': 'fine-tuned (local)', 'local_only': True})
    
    else: # No repo_finetuned specified, use original fallback logic
        logger.info(f"[{model_key_for_log}] No 'repo_finetuned' path specified. Will attempt fallbacks (downloaded, Hugging Face Hub).")
        if repo_downloaded:
            paths_to_try.append({'path': repo_downloaded, 'source': 'downloaded (local)', 'local_only': True})
        if model_name: # Fallback to Hugging Face Hub
            paths_to_try.append({'path': model_name, 'source': 'Hugging Face Hub', 'local_only': False})
            paths_to_try.append({'path': model_name, 'source': 'Hugging Face Hub (local cache attempt)', 'local_only': True})

    if not paths_to_try:
        # This case would occur if repo_finetuned was not specified, and neither repo_downloaded nor model_name were.
        # Or if repo_finetuned was specified but failed checks (already raised RuntimeError).
        logger.error(f"[{model_key_for_log}] No valid paths to attempt loading the model.")
        # Raising an error here ensures that if configuration is incomplete, it's flagged.
        raise RuntimeError(f"No load paths could be determined for model {model_key_for_log} based on configuration and checks.")

    # Determine torch_dtype and load_in_8bit based on flags
    torch_dtype_arg = None
    load_in_8bit_arg = load_in_8bit_override
    
    if not load_in_8bit_arg and use_amp and accelerator.device.type == 'cuda':
        torch_dtype_arg = torch.float16 # For AMP
        logger.info(f"[{model_key_for_log}] AMP enabled, setting torch_dtype to float16 for PyTorch model loading.")
    
    if load_in_8bit_override: # Explicit 8-bit request (e.g., for GPT-Neo on CUDA)
         logger.info(f"[{model_key_for_log}] 8-bit loading is specifically enabled.")


    for config_path_info in paths_to_try:
        path_to_load = config_path_info['path']
        source_info = config_path_info['source']
        local_files_flag = config_path_info['local_only']
        
        logger.info(f"Attempting to load {model_key_for_log} from: {path_to_load} (source: {source_info}, local_files_only={local_files_flag})")
        
        model = None
        tokenizer = None

        try:
            if use_onnxruntime:
                logger.info(f"[{model_key_for_log}] Attempting to load ONNX model from {path_to_load}")
                try:
                    # Optimum expects the directory containing model.onnx
                    # Ensure the onnx model exists at this path or a sub-path
                    # For simplicity, let's assume path_to_load could be an ONNX model directory
                    onnx_model_path = path_to_load 
                    if os.path.exists(os.path.join(onnx_model_path, "model.onnx")): # Check if it's an ONNX dir
                        model = ORTModelForSequenceClassification.from_pretrained(
                            onnx_model_path, 
                            local_files_only=local_files_flag,
                            # provider="CUDAExecutionProvider" # if on GPU, Optimum might handle this
                        )
                        tokenizer = AutoTokenizer.from_pretrained(onnx_model_path, local_files_only=local_files_flag)
                        logger.info(f"Successfully loaded ONNX model {model_key_for_log} from {onnx_model_path}")
                        # ONNX models don't use BetterTransformer or torch_dtype in the same way
                        return model, tokenizer
                    else:
                        logger.warning(f"[{model_key_for_log}] No model.onnx found in {onnx_model_path}. Will try PyTorch model.")
                except Exception as e:
                    logger.warning(f"[{model_key_for_log}] Failed to load ONNX model from {path_to_load}: {e}. Falling back to PyTorch.")
            
            # PyTorch model loading
            model_kwargs = {}
            if torch_dtype_arg and not load_in_8bit_arg : # Don't set torch_dtype if 8-bit
                model_kwargs['torch_dtype'] = torch_dtype_arg
            
            if load_in_8bit_arg: # load_in_8bit_override
                if accelerator.device.type == 'cuda':
                    model_kwargs['load_in_8bit'] = True
                    model_kwargs['device_map'] = 'auto' # Recommended for 8-bit
                else:
                    logger.warning(f"[{model_key_for_log}] 8-bit loading requested but not on CUDA. Loading in default precision.")

            model = AutoModelForSequenceClassification.from_pretrained(path_to_load, local_files_only=local_files_flag, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(path_to_load, local_files_only=local_files_flag)
            logger.info(f"Successfully loaded PyTorch model {model_key_for_log} from {path_to_load} ({source_info})")

            # Apply BetterTransformer if requested and model is PyTorch
            if use_bettertransformer and not load_in_8bit_arg and not isinstance(model, ORTModelForSequenceClassification):
                if accelerator.device.type == 'cuda': # BetterTransformer typically for CUDA
                    try:
                        # Ensure model is on the correct device before transform if not handled by device_map
                        if 'device_map' not in model_kwargs:
                             model = model.to(accelerator.device)
                        model = BetterTransformer.transform(model, keep_original_model=False)
                        logger.info(f"[{model_key_for_log}] Applied BetterTransformer.")
                    except Exception as e:
                        logger.warning(f"[{model_key_for_log}] Could not apply BetterTransformer: {e}")
                else:
                    logger.info(f"[{model_key_for_log}] BetterTransformer requested but not on CUDA, skipping.")
            
            # Device placement for PyTorch models if not handled by device_map (e.g. not 8-bit)
            # Accelerator.prepare() will handle final device placement.
            # Here, we ensure it's on a device if BetterTransformer was applied or for consistency.
            if not isinstance(model, ORTModelForSequenceClassification) and 'device_map' not in model_kwargs:
                 model = model.to(accelerator.device)


            return model, tokenizer
        except OSError:
            logger.warning(f"Failed to load {model_key_for_log} from {path_to_load} ({source_info}) with local_files_only={local_files_flag}. It might not exist or not be a valid model directory.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading {model_key_for_log} from {path_to_load} ({source_info}): {e}")

    logger.error(f"Failed to load model {model_key_for_log} from all specified paths.")
    return None, None