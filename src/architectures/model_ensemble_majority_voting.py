import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
import json
from tqdm.auto import tqdm # Aggiunto import per tqdm
from src.model_configs import MODEL_CONFIGS # Assuming MODEL_CONFIGS is accessible
from src.utils import get_model_type, get_prediction, load_local_model 
from src.data_preprocessing import load_imdb_dataset, create_splits # Moved load_imdb_dataset and create_splits here

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
        self.device = device

        logger.info(f"Initializing EnsembleMajorityVoting with models: {model_keys}")

        for model_key in self.model_keys:
            if model_key not in MODEL_CONFIGS:
                logger.error(f"Configuration for model '{model_key}' not found in MODEL_CONFIGS.")
                raise ValueError(f"Configuration for model '{model_key}' not found.")

            model_config_entry = MODEL_CONFIGS[model_key]
            
            # Use the new load_local_model function
            model, tokenizer = load_local_model(model_config_entry, model_key)

            if model and tokenizer:
                self.models[model_key] = model.to(self.device)
                self.models[model_key].eval() # Set to evaluation mode
                self.tokenizers[model_key] = tokenizer
                self.model_types[model_key] = get_model_type(model_config_entry['model_name'])
                logger.info(f"Successfully loaded and configured model: {model_key}")
            else:
                # This error will be raised if load_local_model returns (None, None)
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
                padding=True,
                max_length=512  # Added max_length for robust padding/truncation
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_batch = outputs.logits # Shape: (batch_size, num_classes)

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

    def evaluate(self, per_device_eval_batch_size=4, output_json_path=None):
        """
        Evaluates the ensemble on the prepared test dataset.

        Args:
            per_device_eval_batch_size (int): Batch size for evaluation.
            output_json_path (str, optional): Path to save detailed predictions and summary.

        Returns:
            dict: A dictionary containing evaluation metrics (e.g., accuracy).
        """
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            logger.error("Test dataset not prepared for Ensemble. Call prepare_datasets() first or check for errors during preparation.")
            raise RuntimeError("Test dataset not prepared for Ensemble. Call prepare_datasets() first.")

        dataset = self.test_dataset # Use the prepared dataset
        correct_predictions = 0
        total_predictions = 0
        all_detailed_predictions = []

        logger.info(f"Starting evaluation of ensemble on dataset with {len(dataset)} samples.")

        # Utilizzo di tqdm per la barra di progresso
        # Calcola il numero totale di batch per tqdm
        num_batches = (len(dataset) + per_device_eval_batch_size - 1) // per_device_eval_batch_size
        
        progress_bar = tqdm(range(num_batches), desc="Evaluating Ensemble")

        for i in progress_bar:
            start_index = i * per_device_eval_batch_size
            end_index = start_index + per_device_eval_batch_size
            
            batch_texts = dataset['text'][start_index:end_index]
            batch_labels = dataset['label'][start_index:end_index]

            if not batch_texts: # Se il batch è vuoto (può succedere all'ultima iterazione se len(dataset) non è multiplo di batch_size)
                continue

            batch_prediction_details = self.predict(batch_texts)

            for idx, detail in enumerate(batch_prediction_details):
                true_label = batch_labels[idx]
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