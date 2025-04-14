# Importa le classi dei modelli specifici.
# Si presume che i file siano nominati come segue:
# - model_bert_base_uncased_imdb.py -> classe BertBaseUncasedIMDB
# - model_bart_base_imdb.py -> classe BartBaseIMDB
# - model_gpt_neo_2_7b_imdb.py -> classe GPTNeo27BIMDB
# - model_ensemble_majority_voting.py -> classe EnsembleMajorityVoting

from architectures.model_bart_base_imdb import BartBaseIMDB
from architectures.model_bert_base_uncased_imdb import BertBaseUncasedIMDB
from architectures.model_ensemble_majority_voting import EnsembleMajorityVoting
from architectures.model_gpt_neo_2_7b_imdb import GPTNeo27BIMDB

def get_model(model_name, **kwargs):
    """
    Restituisce un'istanza del modello in base a 'model_name'.

    Parametri:
      - model_name (str): Uno dei seguenti: 
          'bert-base-uncased-imdb', 
          'bart-base-imdb', 
          'gpt-neo-2.7b-imdb', 
          'ensemble_majority_voting'
      - kwargs: Parametri addizionali per la costruzione del modello.

    Ritorna:
      - Un'istanza della classe modello corrispondente, inizializzata con il repository
        Hugging Face associato (se necessario) e altri parametri extra.
    """
    # Mappa dei nomi dei modelli alle rispettive classi
    model_map = {
        "bert-base-uncased-imdb": BertBaseUncasedIMDB,
        "bart-base-imdb": BartBaseIMDB,
        "gpt-neo-2.7b-imdb": GPTNeo27BIMDB,
        "ensemble_majority_voting": EnsembleMajorityVoting,
    }

    if model_name not in model_map:
        raise ValueError(
            f"Modello sconosciuto: {model_name}. I modelli disponibili sono: {list(model_map.keys())}"
        )

    # Recupera il repository associato al modello (se occorre)
    repo = MODELS_TO_UPLOAD.get(model_name)
    # Instanzia il modello passando il repository ed eventuali altri parametri
    return model_map[model_name](repo=repo, **kwargs)
