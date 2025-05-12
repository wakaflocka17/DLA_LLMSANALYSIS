MODEL_CONFIGS = {
    "bert_base_uncased": {
        "model_name": "bert-base-uncased",
        "repo_pretrained": "bert-base-uncased", # HF Hub name
        "repo_finetuned": "models/finetuned/bert-base-uncased-imdb", # Local path to fine-tuned version
        "repo_downloaded": "models/downloaded/bert_base_uncased", # Local path to downloaded HF model
        "epochs": 3,
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "num_labels": 2,
    },
    "bart_base": {
        "model_name": "facebook/bart-base",
        "repo_pretrained": "facebook/bart-base",
        "repo_finetuned": "models/finetuned/bart-base-imdb",
        "repo_downloaded": "models/downloaded/bart_base",
        "epochs": 3,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "num_labels": 2,
    },
    "gpt_neo_2_7b": {
        "model_name": "EleutherAI/gpt-neo-2.7B",
        "repo_pretrained": "EleutherAI/gpt-neo-2.7B",
        "repo_finetuned": "models/finetuned/gpt-neo-2.7B-imdb",
        "repo_downloaded": "models/downloaded/gpt_neo_2_7b",
        "epochs": 1, # Example
        "train_batch_size": 1, # Example
        "eval_batch_size": 1,  # Example
        "num_labels": 2,
    },
    "ensemble_majority_voting": {
        "model_names": ["bert_base_uncased", "bart_base"], # Add all model keys part of the ensemble
        "repo": "models/ensemble_majority_voting", # This is the directory where the ensemble will be saved
        "eval_batch_size": 64, # Example, can be overridden
        # Add other ensemble specific configs if any
    }
    # Add other model configurations as needed
}