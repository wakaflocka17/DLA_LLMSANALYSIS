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
    def __init__(self, model_keys, accelerator, use_amp, use_bettertransformer, use_onnxruntime):
        """
        Initializes the ensemble model with majority voting.

        Args:
            model_keys (list): List of model keys from MODEL_CONFIGS.
            accelerator (Accelerator): The Hugging Face Accelerator object.
            use_amp (bool): Whether Automatic Mixed Precision is enabled.
            use_bettertransformer (bool): Whether to apply BetterTransformer.
            use_onnxruntime (bool): Whether to use ONNX Runtime for inference.
        """
        self.model_keys = model_keys
        self.accelerator = accelerator
        self.device = self.accelerator.device # Main device from Accelerator
        self.use_amp = use_amp # Store for reference, though Accelerator manages AMP
        self.use_bettertransformer = use_bettertransformer
        self.use_onnxruntime = use_onnxruntime

        self.models = {}
        self.tokenizers = {}
        self.model_types = {} # To store original model types ('encoder-only', etc.)
        self.is_onnx_model = {} # To track if a model is ONNX

        logger.info(f"Initializing EnsembleMajorityVoting with models: {model_keys} on device: {self.device}")
        logger.info(f"AMP: {self.use_amp}, BetterTransformer: {self.use_bettertransformer}, ONNXRuntime: {self.use_onnxruntime}")

        for model_key in self.model_keys:
            if model_key not in MODEL_CONFIGS:
                logger.error(f"Configuration for model '{model_key}' not found in MODEL_CONFIGS.")
                raise ValueError(f"Configuration for model '{model_key}' not found.")

            model_config_entry = MODEL_CONFIGS[model_key]
            
            # Specific 8-bit handling for GPT-Neo as per original logic
            load_in_8bit_override_flag = False
            if "gpt" in model_key.lower() and self.device.type == 'cuda':
                load_in_8bit_override_flag = True
                logger.info(f"Configuring {model_key} for 8-bit quantization (override).")


            model, tokenizer = load_local_model(
                model_config_entry,
                model_key,
                accelerator=self.accelerator,
                use_amp=self.use_amp,
                use_bettertransformer=self.use_bettertransformer, # Pass this down
                use_onnxruntime=self.use_onnxruntime, # Pass this down
                load_in_8bit_override=load_in_8bit_override_flag
            )

            if model and tokenizer:
                # Check if the loaded model is an ONNX model
                # ORTModelForSequenceClassification is not imported here, so check by type name string
                is_onnx = "ORTModel" in str(type(model)) 
                self.is_onnx_model[model_key] = is_onnx

                if not is_onnx: # PyTorch model
                    # Accelerator prepares PyTorch models (handles device placement, AMP wrapping)
                    # 8-bit models with device_map='auto' are handled correctly by Accelerator.
                    # Non-8-bit models are moved to accelerator.device by load_local_model or here before prepare.
                    if not load_in_8bit_override_flag: # if not 8-bit with device_map
                         model = model.to(self.accelerator.device)
                    
                    prepared_model = self.accelerator.prepare(model)
                    self.models[model_key] = prepared_model
                else: # ONNX model
                    # ONNX models are not prepared by Accelerator in the same way.
                    # Device placement for ONNX is usually handled by the ONNX Runtime provider.
                    # We assume load_local_model configured it for CUDA if available.
                    self.models[model_key] = model # Store the ONNX model directly

                self.tokenizers[model_key] = tokenizer
                self.model_types[model_key] = get_model_type(model_config_entry['model_name'])
                
                # Set to eval mode
                if hasattr(self.models[model_key], 'eval'):
                    self.models[model_key].eval()

                logger.info(f"Successfully loaded and configured model: {model_key} (ONNX: {is_onnx})")
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
        Accelerator handles mixed precision for prepared PyTorch models.
        """
        model_batch_predictions = {model_key: [] for model_key in self.model_keys}

        for model_key in self.model_keys:
            model = self.models[model_key]
            tokenizer = self.tokenizers[model_key]
            
            inputs = tokenizer(
                text_batch,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512 # Consistent max length
            )
            
            # Move inputs to the device of the model (especially important for ONNX or non-prepared models)
            # For prepared PyTorch models, accelerator.device is the target.
            # For ONNX, it might be CPU or CUDA depending on ORT setup.
            # Safest to move to model.device if available, else accelerator.device.
            target_device = model.device if hasattr(model, 'device') else self.accelerator.device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.no_grad():
                # For PyTorch models prepared by Accelerator with AMP, autocast is handled by Accelerator.
                # For ONNX models, torch.amp.autocast is not applicable.
                # For 8-bit PyTorch models, Accelerator also handles them correctly without explicit fp16 casting.
                
                # The explicit torch.amp.autocast block from original code is removed here,
                # relying on Accelerator for PyTorch models.
                # If a PyTorch model was NOT prepared by Accelerator (e.g. an error or different setup),
                # and AMP is desired, manual autocast would be needed.
                # Current setup assumes all PyTorch models are prepared.

                outputs = model(**inputs)
                logits_batch = outputs.logits

            batch_preds = logits_batch.argmax(dim=-1).tolist()
            model_batch_predictions[model_key] = batch_preds
        
        # Now, assemble the results for each item in the original batch
        batch_size = len(text_batch)
        final_batch_predictions_details = []

        for i in range(batch_size): 
            votes = []
            individual_model_preds_for_item = {}
            for model_key in self.model_keys: 
                prediction_for_item = model_batch_predictions[model_key][i]
                individual_model_preds_for_item[model_key] = prediction_for_item
                votes.append(prediction_for_item)
            
            if votes:
                final_vote = max(set(votes), key=votes.count)
            else:
                final_vote = -1 
                logger.warning(f"No votes collected for item index {i} in the current batch.")
            
            prediction_detail = {**individual_model_preds_for_item, "voted": final_vote}
            final_batch_predictions_details.append(prediction_detail)
            
        return final_batch_predictions_details

    def evaluate(self, per_device_eval_batch_size=64, output_json_path=None): # Default batch size updated
        """
        Evaluates the ensemble on the prepared test dataset.
        Uses Accelerator for DataLoader preparation.
        """
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            logger.error("Test dataset not prepared. Call prepare_datasets() first.")
            raise RuntimeError("Test dataset not prepared.")

        dataset = self.test_dataset
        
        # DataLoader tuning
        num_workers = 0
        pin_memory_flag = False
        if self.accelerator.device.type == 'cuda':
            try:
                num_cuda_devices = torch.cuda.device_count()
                num_workers = 4 * num_cuda_devices 
                pin_memory_flag = True
                logger.info(f"CUDA detected. DataLoader: num_workers={num_workers}, pin_memory=True")
            except Exception as e:
                logger.warning(f"Could not get CUDA device count, defaulting num_workers to 0: {e}")
        else:
            logger.info(f"Non-CUDA device ({self.accelerator.device.type}). DataLoader: num_workers=0, pin_memory=False")

        eval_dataloader = DataLoader(
            dataset,
            batch_size=per_device_eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory_flag,
            collate_fn=None # Use default collate_fn from HF datasets
        )

        # Prepare DataLoader with Accelerator
        prepared_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # Models are already prepared in __init__

        correct_predictions = 0
        total_predictions = 0
        all_detailed_predictions = []

        logger.info(f"Starting evaluation of ensemble on {len(dataset)} samples with batch size {per_device_eval_batch_size}.")
        
        progress_bar = tqdm(prepared_dataloader, desc="Evaluating Ensemble", disable=not self.accelerator.is_local_main_process)

        for batch in progress_bar:
            # Batch already on correct device due to accelerator.prepare(dataloader)
            batch_texts = batch['text'] # Assuming 'text' and 'label' keys
            batch_labels = batch['label']

            if not batch_texts:
                continue

            batch_prediction_details = self.predict(batch_texts) # predict expects list of strings

            # Gather predictions if distributed (Accelerator handles this if needed, but predict is per-process)
            # For simple majority voting evaluation, direct processing is fine.

            for idx, detail in enumerate(batch_prediction_details):
                # Ensure batch_labels are correctly indexed and converted if they are tensors
                true_label_tensor_or_val = batch_labels[idx]
                true_label = true_label_tensor_or_val.item() if isinstance(true_label_tensor_or_val, torch.Tensor) else true_label_tensor_or_val
                
                ensemble_vote = detail['voted']
                
                if ensemble_vote == true_label:
                    correct_predictions += 1
                total_predictions += 1
                
                # Store all details, including true label
                all_detailed_predictions.append({
                    "text_sample": batch_texts[idx][:100] + "..." if isinstance(batch_texts[idx], str) else "N/A", # Log a snippet
                    **detail, 
                    "true_label": true_label
                })
        
        # Aggregate results if using distributed evaluation (more complex, for now assume single process or simple sum)
        # If using accelerator.gather_for_metrics, results would need to be handled carefully.
        # For this task, simple aggregation is shown.
        
        # Note: For multi-GPU, correct_predictions and total_predictions would need to be summed across processes.
        # accelerator.reduce for sum can be used here.
        # E.g., total_correct_t = torch.tensor(correct_predictions).to(self.accelerator.device)
        # gathered_correct = self.accelerator.gather(total_correct_t)
        # final_correct = torch.sum(gathered_correct).item() ... and similarly for total_predictions.
        # For simplicity, this example assumes single-process or that this logic is sufficient for the setup.


        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        if self.accelerator.is_local_main_process: # Logging and saving only on main process
            logger.info(f"Ensemble Evaluation Complete. Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")

            results = {
                "accuracy": accuracy,
                "total_samples": total_predictions,
                "correct_predictions": correct_predictions,
            }

            if output_json_path:
                # Ensure directory exists
                output_dir_path = os.path.dirname(output_json_path)
                if output_dir_path: # Check if output_dir is not an empty string
                    os.makedirs(output_dir_path, exist_ok=True)

                # Save all detailed predictions for analysis from the main process
                results_to_save = {**results, "detailed_predictions_main_process": all_detailed_predictions}
                try:
                    with open(output_json_path, 'w') as f:
                        json.dump(results_to_save, f, indent=4)
                    logger.info(f"Ensemble evaluation results saved to {output_json_path}")
                except Exception as e:
                    logger.error(f"Failed to save ensemble results to {output_json_path}: {e}")
            return results
        return {} # Other processes return empty or None

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