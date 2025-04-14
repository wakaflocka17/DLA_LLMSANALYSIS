import logging
import os
import sys
from importlib import import_module

# Add the project root directory to sys.path to enable imports from architectures
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        'bart_base': ('architectures.model_bart_base_imdb', 'BartBaseIMDB'),
        'bert_base_uncased': ('architectures.model_bert_base_uncased_imdb', 'BertBaseUncasedIMDB'),
        'gpt_neo_2_7b': ('architectures.model_gpt_neo_2_7b_imdb', 'GPTNeo27BIMDB'),
        'ensemble_majority_voting': ('architectures.model_ensemble_majority_voting', 'EnsembleMajorityVoting')
    }
    
    if model_config_key not in model_mapping:
        raise ValueError(f"Unknown model configuration key: {model_config_key}")
    
    module_path, class_name = model_mapping[model_config_key]
    
    try:
        # Dynamically import the module and get the class
        module = import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Create and return an instance of the model
        return model_class()
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import model {model_config_key}: {e}")
        raise
