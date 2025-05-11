from importlib import import_module
from src.model_configs import MODEL_CONFIGS
import logging

logger = logging.getLogger(__name__)

def get_model(model_config_key, use_downloaded: bool = False, **kwargs_for_model_init):
    """
    Factory function to create model instances based on configuration key.

    Args:
        model_config_key: String key matching a model in MODEL_CONFIGS.
        use_downloaded: If True, instructs individual models (or ensemble members)
                        to attempt loading from 'repo_downloaded' path first.
        **kwargs_for_model_init: Additional keyword arguments to pass to the model's constructor.

    Returns:
        An instance of the appropriate model class.
    """
    model_mapping = {
        'bart_base': ('src.architectures.model_bart_base_imdb', 'BartBaseIMDB'),
        'bert_base_uncased': ('src.architectures.model_bert_base_uncased_imdb', 'BertBaseUncasedIMDB'),
        'gpt_neo_2_7b': ('src.architectures.model_gpt_neo_2_7b_imdb', 'GPTNeo27BIMDB'),
        'ensemble_majority_voting': ('src.architectures.model_ensemble_majority_voting', 'EnsembleMajorityVoting')
    }

    if model_config_key not in model_mapping:
        raise ValueError(f"Unknown model configuration key: {model_config_key}")

    # Ottieni la configurazione specifica
    config = MODEL_CONFIGS.get(model_config_key, {})

    module_path, class_name = model_mapping[model_config_key]

    try:
        # Importa dinamicamente il modulo e ottieni la classe
        module = import_module(module_path)
        model_class = getattr(module, class_name)

        # Gestione percorsi
        if use_downloaded:
            repo_finetuned = config.get('repo_downloaded', config.get('repo_finetuned'))
        else:
            repo_finetuned = config.get('repo_finetuned', config.get('repo'))

        repo_pretrained = config.get('repo_pretrained', config.get('repo'))

        # Se il modello è l'ensemble, ha bisogno solo di 'repo'
        if model_config_key == "ensemble_majority_voting":
            return model_class(repo=config.get('repo'), use_downloaded=use_downloaded)

        return model_class(repo_finetuned=repo_finetuned, repo_pretrained=repo_pretrained)

    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import model {model_config_key}: {e}")
        raise