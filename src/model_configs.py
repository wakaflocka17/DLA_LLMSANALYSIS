MODEL_CONFIGS = {
    'bart_base': {
        'model_name': 'facebook/bart-base',
        'epochs': 5,
        'train_batch_size': 16,
        'eval_batch_size': 32,
        'learning_rate': 2e-5,
        'repo': 'models/bart-base',
    },
    'bert_base_uncased': {
        'model_name': 'google-bert/bert-base-uncased',
        'epochs': 5,
        'train_batch_size': 16,
        'eval_batch_size': 32,
        'learning_rate': 2e-5,
        'repo': 'models/bert_base_uncased',
    },
    'gpt_neo_2_7b': {
        'model_name': 'EleutherAI/gpt-neo-2.7B',
        'epochs': 3,
        'train_batch_size': 4,
        'eval_batch_size': 4,  # Potrebbe rimanere basso per ragioni di memoria
        'learning_rate': 1e-5,
        'gradient_accumulation_steps': 4,
        'repo': 'models/gpt-neo-2.7B',
    },
    'ensemble_majority_voting': {
        'model_names': ['facebook/bart-base', 'bert-base-uncased', 'EleutherAI/gpt-neo-2.7B'],
        'train_batch_size': 8,
        'eval_batch_size': 16,
        'epochs': 3,
        'repo': 'models/majority-voting',
    }
}