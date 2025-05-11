import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
import json
from tqdm.auto import tqdm # Aggiunto import per tqdm
from src.model_configs import MODEL_CONFIGS # Assuming MODEL_CONFIGS is accessible
from src.utils import get_model_type, load_local_model # Removed get_prediction as it's not used here
from src.data_preprocessing import load_imdb_dataset, create_splits # Moved load_imdb_dataset and create_splits here
from torch.utils.data import DataLoader # Added DataLoader
from optimum.bettertransformer import BetterTransformer # Added BetterTransformer

logger = logging.getLogger(__name__)

class EnsembleMajorityVoting:
    def __init__(self, model_keys, device='cpu'):
        """
        Initializes the ensemble model with majority voting.

        Args:
            model_keys (list): List of model keys (e.g., ['bart_base', 'bert_base_uncased'])
                               as defined in MODEL_CONFIGS.
            device (str): Device to run inference on ('cpu' or 'cuda').
        """
        self.model_keys = model_keys
        self.models = {}
        self.tokenizers = {}
        self.model_types = {}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu") # Ensure device is torch.device

        logger.info(f"Initializing EnsembleMajorityVoting with models: {model_keys} on device: {self.device}")

        for model_key in self.model_keys:
            if model_key not in MODEL_CONFIGS:
                logger.error(f"Configuration for model '{model_key}' not found in MODEL_CONFIGS.")
                raise ValueError(f"Configuration for model '{model_key}' not found.")

            model_config_entry = MODEL_CONFIGS[model_key]
            model_name_for_type = model_config_entry['model_name'] # For get_model_type
            
            current_torch_dtype = None
            current_load_in_8bit = False

            if "gpt" in model_key.lower(): # Specific handling for GPT-Neo
                if self.device.type == 'cuda':
                    current_load_in_8bit = True
                    logger.info(f"Configuring {model_key} for 8-bit quantization.")
                else:
                    logger.warning(f"8-bit quantization for {model_key} is only supported on CUDA. Loading in default precision on CPU.")
            elif self.device.type == 'cuda': # For BART and BERT on CUDA
                current_torch_dtype = torch.bfloat16 # A100 supports bfloat16
                logger.info(f"Configuring {model_key} with dtype {current_torch_dtype}.")

            model, tokenizer = load_local_model(
                model_config_entry, 
                model_key,
                torch_dtype=current_torch_dtype,
                load_in_8bit=current_load_in_8bit
            )

            if model and tokenizer:
                if not current_load_in_8bit: # If not 8-bit, move to device manually
                    model = model.to(self.device)
                
                # Apply BetterTransformer if applicable (BART, BERT) and on CUDA
                model_arch_type = get_model_type(model_name_for_type)
                if model_arch_type in ["encoder-only", "encoder-decoder"] and "gpt" not in model_key.lower() and self.device.type == 'cuda':
                    try:
                        model = BetterTransformer.transform(model)
                        logger.info(f"Applied BetterTransformer to {model_key}.")
                    except Exception as e:
                        logger.warning(f"Could not apply BetterTransformer to {model_key}: {e}")
                
                model.eval() # Set to evaluation mode
                self.models[model_key] = model
                self.tokenizers[model_key] = tokenizer
                self.model_types[model_key] = get_model_type(model_config_entry['model_name'])
                logger.info(f"Successfully loaded and configured model: {model_key}")
            else:
                logger.error(f"Failed to load model and tokenizer for {model_key}. Ensemble cannot be initialized.")
                raise RuntimeError(f"Failed to load model and tokenizer for {model_key}.")
        
        logger.info("EnsembleMajorityVoting initialized successfully.")

    def prepare_datasets(self, max_samples: int = None):
        """
        Prepara i dataset per l'evaluation dell'ensemble.
        Carica il dataset IMDB e applica uno split, memorizzando il test_dataset.
        """
        logger.info("Preparing dataset for EnsembleMajorityVoting (loading IMDB).")
        try:
            full_dataset = load_imdb_dataset()
            # Assumiamo che create_splits restituisca train, validation, test
            # Per l'ensemble, siamo interessati principalmente al test_dataset per la valutazione.
            # Se create_splits ha una firma diversa, questo potrebbe necessitare di aggiustamenti.
            _train_dataset, _val_dataset, self.test_dataset = create_splits(full_dataset)
        except Exception as e:
            logger.error(f"Failed to load or split dataset for Ensemble: {e}")
            self.test_dataset = None
            # Potrebbe essere opportuno sollevare l'eccezione se il caricamento del dataset è critico
            # raise RuntimeError(f"Failed to load or split dataset for Ensemble: {e}") from e
            return

        if self.test_dataset and max_samples is not None:
            if len(self.test_dataset) > max_samples:
                try:
                    self.test_dataset = self.test_dataset.shuffle(seed=42).select(range(max_samples))
                    logger.info(f"Using a subset of {len(self.test_dataset)} samples for evaluation.")
                except Exception as e:
                    logger.error(f"Failed to select subset for test_dataset: {e}")
            else:
                logger.info(f"max_samples ({max_samples}) is >= dataset size ({len(self.test_dataset)}). Using full test dataset.")
        
        if not self.test_dataset:
             logger.warning("Test dataset could not be prepared or is empty for EnsembleMajorityVoting.")
        else:
            logger.info(f"Dataset prepared for EnsembleMajorityVoting. Test set size: {len(self.test_dataset)}")

    def predict(self, text_batch: list[str]):
        """
        Makes predictions for a batch of texts using the ensemble.
        Processes the entire batch for each model to improve efficiency.

        Args:
            text_batch (list of str): A list of input texts.

        Returns:
            list of dict: A list of prediction details for each text, including
                          individual model votes and the final ensemble vote.
        """
        # Store all predictions from all models for the batch
        # Example: model_batch_predictions['bart_base'] = [pred0, pred1, pred2, pred3] for a batch of 4
        model_batch_predictions = {model_key: [] for model_key in self.model_keys}

        for model_key in self.model_keys:
            model = self.models[model_key]
            tokenizer = self.tokenizers[model_key]
            # model_type = self.model_types[model_key] # Not directly used for argmax with AutoModelForSequenceClassification

            # Tokenize the whole batch of texts
            # Using a common max_length like 512 for consistency, truncation handles longer texts.
            inputs = tokenizer(
                text_batch,
                return_tensors="pt",
                truncation=True,
                padding="max_length", # Changed to max_length for consistency, helps with static shapes for compiled models
                max_length=512
            ).to(self.models[model_key].device) # Ensure inputs are on the same device as the model
            
            with torch.no_grad():
                # For bfloat16/float16 on CUDA, autocast can be beneficial if not using torch_dtype in from_pretrained
                # However, with torch_dtype set at loading, it's often handled.
                # For 8-bit, autocast is not typically needed.
                # Adding it for safety for bfloat16 models if mixed precision parts exist.
                cast_dtype = torch.bfloat16 if self.models[model_key].dtype == torch.bfloat16 else torch.float16
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda' and not self.models[model_key].config.quantization_config.load_in_8bit), dtype=cast_dtype):
                    outputs = model(**inputs)
                logits_batch = outputs.logits

            # Get predictions for the entire batch
            # For AutoModelForSequenceClassification, logits are (batch_size, num_classes)
            # .tolist() converts the tensor of predictions to a Python list of integers
            batch_preds = logits_batch.argmax(dim=-1).tolist()
            model_batch_predictions[model_key] = batch_preds

        # Now, assemble the results for each item in the original batch
        batch_size = len(text_batch)
        final_batch_predictions_details = []

        for i in range(batch_size): # Iterate through each item in the batch
            votes = []
            individual_model_preds_for_item = {}
            for model_key in self.model_keys: # Collect votes for this item from all models
                prediction_for_item = model_batch_predictions[model_key][i]
                individual_model_preds_for_item[model_key] = prediction_for_item
                votes.append(prediction_for_item)
            
            # Majority voting for the current item
            if votes:
                final_vote = max(set(votes), key=votes.count)
            else:
                # This case should ideally not be reached if models are making predictions
                final_vote = -1 
                logger.warning(f"No votes collected for item index {i} in the current batch.")
            
            prediction_detail = {**individual_model_preds_for_item, "voted": final_vote}
            final_batch_predictions_details.append(prediction_detail)
            
        return final_batch_predictions_details

    def evaluate(self, per_device_eval_batch_size=4, output_json_path=None, num_dataloader_workers=4):
        """
        Evaluates the ensemble on the prepared test dataset.

        Args:
            per_device_eval_batch_size (int): Batch size for evaluation.
            output_json_path (str, optional): Path to save detailed predictions and summary.
            num_dataloader_workers (int): Number of workers for DataLoader.

        Returns:
            dict: A dictionary containing evaluation metrics (e.g., accuracy).
        """
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            logger.error("Test dataset not prepared for Ensemble. Call prepare_datasets() first or check for errors during preparation.")
            raise RuntimeError("Test dataset not prepared for Ensemble. Call prepare_datasets() first.")

        dataset = self.test_dataset
        correct_predictions = 0
        total_predictions = 0
        all_detailed_predictions = []

        logger.info(f"Starting evaluation of ensemble on dataset with {len(dataset)} samples.")
        
        # Use PyTorch DataLoader
        # Ensure your self.test_dataset is compatible (Hugging Face datasets are)
        # If it's a custom list of dicts, you might need a simple custom Dataset class
        eval_dataloader = DataLoader(
            dataset,
            batch_size=per_device_eval_batch_size,
            shuffle=False, # No need to shuffle for evaluation
            num_workers=num_dataloader_workers if self.device.type == 'cuda' else 0, # num_workers > 0 can cause issues on CPU with some setups
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        progress_bar = tqdm(eval_dataloader, desc="Evaluating Ensemble")

        for batch in progress_bar:
            # Assuming batch is a dictionary from Hugging Face dataset, containing 'text' and 'label'
            batch_texts = batch['text']
            batch_labels = batch['label'] # These are Python integers or list of integers

            if not batch_texts:
                continue

            batch_prediction_details = self.predict(batch_texts)

            for idx, detail in enumerate(batch_prediction_details):
                true_label = batch_labels[idx].item() if isinstance(batch_labels[idx], torch.Tensor) else batch_labels[idx]
                ensemble_vote = detail['voted']
                
                if ensemble_vote == true_label:
                    correct_predictions += 1
                total_predictions += 1
                
                all_detailed_predictions.append({**detail, "true_label": true_label})

            # Aggiorna la descrizione della progress bar (opzionale, ma può essere utile)
            # if total_predictions > 0:
            #     current_accuracy = correct_predictions / total_predictions
            #     progress_bar.set_postfix({"Acc": f"{current_accuracy:.3f}"})


        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logger.info(f"Ensemble Evaluation Complete. Accuracy: {accuracy:.4f}")

        results = {
            "accuracy": accuracy,
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
        }

        if output_json_path:
            results_to_save = {**results, "detailed_predictions": all_detailed_predictions}
            try:
                # Ensure the directory exists
                output_dir = os.path.dirname(output_json_path)
                if output_dir: # Check if output_dir is not an empty string (e.g. if path is just a filename)
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(output_json_path, 'w') as f:
                    json.dump(results_to_save, f, indent=4)
                logger.info(f"Ensemble evaluation results saved to {output_json_path}")
            except Exception as e:
                logger.error(f"Failed to save ensemble results to {output_json_path}: {e}")
        
        return results

# Example of how it might be called from evaluate_ensemble.py
# This is illustrative and depends on the structure of evaluate_ensemble.py
if __name__ == '__main__':
    # This is a placeholder for actual dataset loading and setup
    # from datasets import load_dataset
    # dummy_dataset = load_dataset("imdb", split="test[:1%]") # Small sample for testing
    
    # ensemble_cfg = MODEL_CONFIGS['ensemble_majority_voting']
    # ensemble = EnsembleMajorityVoting(model_keys=ensemble_cfg['model_names'], device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # # Example text batch
    # texts = ["This movie was fantastic!", "I absolutely hated this film.", "It was an okay movie, not great but not terrible."]
    # predictions = ensemble.predict(texts)
    # for i, p in enumerate(predictions):
    #     logger.info(f"Text: '{texts[i]}' -> Prediction: {p}")

    # # Example evaluation
    # # eval_results = ensemble.evaluate(dummy_dataset, batch_size=4, output_json_path="results/ensemble_eval_details.json")
    # # logger.info(f"Evaluation results: {eval_results}")
    pass