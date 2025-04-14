# Configuration settings for different models
MODEL_CONFIGS = {
    'bart_base': {
        'model_name': 'facebook/bart-base',
        'epochs': 2,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'repo': 'facebook/bart-base',  # Added repo parameter
    },
    'bert_base_uncased': {
        'model_name': 'bert-base-uncased',
        'epochs': 2,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'repo': 'google-bert/bert-base-uncased',
    },
    'gpt_neo_2_7b': {
        'model_name': 'EleutherAI/gpt-neo-2.7B',
        'epochs': 2,
        'batch_size': 8,  # Smaller batch size for large model
        'learning_rate': 1e-5,
        'gradient_accumulation_steps': 4,
        'repo': 'EleutherAI/gpt-neo-2.7B',  # Added repo parameter
    },
    'ensemble_majority_voting': {
        'model_names': ['facebook/bart-base', 'bert-base-uncased', 'EleutherAI/gpt-neo-2.7B'],
        'batch_size': 8,
        'repo': 'ensemble/majority-voting',  # Added repo parameter
    }
}