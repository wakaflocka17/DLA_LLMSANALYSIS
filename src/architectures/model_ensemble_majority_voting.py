import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from ..model_configs import MODEL_CONFIGS # Assuming MODEL_CONFIGS is accessible
from ..utils import get_model_type, get_prediction, load_local_model # Import the new loading function

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

    def predict(self, text_batch):
        """
        Makes predictions for a batch of texts using the ensemble.

        Args:
            text_batch (list of str): A list of input texts.

        Returns:
            list of dict: A list of prediction details for each text, including
                          individual model votes and the final ensemble vote.
                          Example: [{'bart_base': 0, 'bert_base_uncased': 1, 'voted': 0, 'true_label': (optional)}]
        """
        batch_predictions_details = []

        for text_idx, text in enumerate(text_batch):
            votes = []
            individual_model_preds = {}

            for model_key in self.model_keys:
                model = self.models[model_key]
                tokenizer = self.tokenizers[model_key]
                model_type = self.model_types[model_key]

                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prediction = get_prediction(logits, model_type)
                
                votes.append(prediction)
                individual_model_preds[model_key] = prediction
            
            # Majority voting
            # This simple version takes the most common vote.
            # More sophisticated tie-breaking can be added if needed.
            if votes:
                final_vote = max(set(votes), key=votes.count)
            else:
                final_vote = -1 # Or some other indicator of failure/no votes
                logger.warning(f"No votes collected for text index {text_idx}: '{text[:50]}...'")


            prediction_detail = {**individual_model_preds, "voted": final_vote}
            batch_predictions_details.append(prediction_detail)
            
        return batch_predictions_details

    def evaluate(self, dataset, batch_size=4, output_json_path=None):
        """
        Evaluates the ensemble on a given dataset.

        Args:
            dataset (Dataset): A Hugging Face Dataset object with 'text' and 'label' columns.
            batch_size (int): Batch size for evaluation.
            output_json_path (str, optional): Path to save detailed predictions and summary.

        Returns:
            dict: A dictionary containing evaluation metrics (e.g., accuracy).
        """
        correct_predictions = 0
        total_predictions = 0
        all_detailed_predictions = []

        logger.info(f"Starting evaluation of ensemble on dataset with {len(dataset)} samples.")

        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset['text'][i:i+batch_size]
            batch_labels = dataset['label'][i:i+batch_size]

            # Get predictions from the ensemble
            # The predict method now returns a list of dicts, one for each item in batch_texts
            batch_prediction_details = self.predict(batch_texts)

            for idx, detail in enumerate(batch_prediction_details):
                true_label = batch_labels[idx]
                ensemble_vote = detail['voted']
                
                if ensemble_vote == true_label:
                    correct_predictions += 1
                total_predictions += 1
                
                # Store detailed prediction including true label
                all_detailed_predictions.append({**detail, "true_label": true_label})

            if (i // batch_size) % 10 == 0: # Log progress every 10 batches
                 logger.info(f"Processed {i + len(batch_texts)} / {len(dataset)} samples...")


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
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
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