from importlib import import_module

import logging
logger = logging.getLogger(__name__)

def get_model(model_config_key):
    """
    Factory function to create model instances based on configuration key.
    
    Args:
        model_config_key: String key matching a model in MODEL_CONFIGS
        
    Returns:
        An instance of the appropriate model class
    """
    # Map config keys to module and class names
    model_mapping = {
        'bart_base': ('src.architectures.model_bart_base_imdb', 'BartBaseIMDB'),
        'bert_base_uncased': ('src.architectures.model_bert_base_uncased_imdb', 'BertBaseUncasedIMDB'),
        'gpt_neo_2_7b': ('src.architectures.model_gpt_neo_2_7b_imdb', 'GPTNeo27BIMDB'),
        'ensemble_majority_voting': ('src.architectures.model_ensemble_majority_voting', 'EnsembleMajorityVoting')
    }
    
    if model_config_key not in model_mapping:
        raise ValueError(f"Unknown model configuration key: {model_config_key}")
    
    # Ottieni la configurazione del modello
    from src.model_configs import MODEL_CONFIGS
    config = MODEL_CONFIGS.get(model_config_key, {})
    
    module_path, class_name = model_mapping[model_config_key]
    
    try:
        # Importa dinamicamente il modulo e ottieni la classe
        module = import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Se sono definiti, leggi i repository separati per il modello pre-addestrato e quello fine-tunato.
        repo_finetuned = config.get('repo_finetuned', config.get('repo', f'models/{model_config_key}_finetuned'))
        repo_pretrained = config.get('repo_pretrained', config.get('repo', f'models/{model_config_key}_pretrained'))
        
        # Se il modello è l'ensemble, ha bisogno solo di 'repo'
        if model_config_key == "ensemble_majority_voting":
            return model_class(repo=config.get('repo'))

        # Altrimenti, gestiamo i modelli classici
        return model_class(repo_finetuned=repo_finetuned, repo_pretrained=repo_pretrained)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import model {model_config_key}: {e}")
        raise