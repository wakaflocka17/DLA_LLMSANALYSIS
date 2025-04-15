MODEL_CONFIGS = {
    'bart_base': {
        'model_name': 'facebook/bart-base',
        'epochs': 5,
        'train_batch_size': 8,
        'eval_batch_size': 4,
        'learning_rate': 2e-5,
        'repo_pretrained': 'models/pretrained/bart-base',
        'repo_finetuned': 'models/finetuned/bart-base-imdb',
    },
    'bert_base_uncased': {
        'model_name': 'google-bert/bert-base-uncased',
        'epochs': 5,
        'train_batch_size': 8,
        'eval_batch_size': 4,
        'learning_rate': 2e-5,
        'repo_pretrained': 'models/pretrained/bert-base-uncased',
        'repo_finetuned': 'models/finetuned/bert-base-uncased-imdb',
    },
    'gpt_neo_2_7b': {
        'model_name': 'EleutherAI/gpt-neo-2.7B',
        'epochs': 3,
        'train_batch_size': 1,
        'eval_batch_size': 1,
        'learning_rate': 1e-5,
        'gradient_accumulation_steps': 8,
        'repo_pretrained': 'models/pretrained/gpt-neo-2.7B',
        'repo_finetuned': 'models/finetuned/gpt-neo-2.7B-imdb',
    },
    'ensemble_majority_voting': {
        'model_names': ['facebook/bart-base', 'bert-base-uncased', 'EleutherAI/gpt-neo-2.7B'],
        'train_batch_size': 8,
        'eval_batch_size': 4,
        'epochs': 3,
        'repo': 'models/ensemble/majority-voting-imdb'
    }
}