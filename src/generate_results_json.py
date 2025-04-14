import json
import argparse
import logging
from model_factory import get_model
from src.data_preprocessing import load_imdb_dataset, create_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_models(model_names, max_samples, batch_size, output_file):
    """
    Valuta i modelli indicati e salva i risultati in un file JSON.
    Parametri:
      - model_names: lista di nomi dei modelli (es. 'bart-base-imdb', 'bert-base-uncased-imdb', etc.)
      - max_samples: numero massimo di campioni da usare (utile per debug)
      - batch_size: batch size da usare per la valutazione
      - output_file: nome del file JSON di output
    """
    results = {}

    # Carica il dataset IMDB una volta sola
    logging.info("Loading dataset...")
    dataset = load_imdb_dataset()
    _, _, test_data = create_splits(dataset)

    for model_name in model_names:
        logging.info(f"Evaluating model: {model_name}")
        try:
            model = get_model(model_name)
            # Prepara i dataset, facendoli elaborare (tokenizzazione, ecc.)
            model.prepare_datasets(max_samples=max_samples)
            # Esegue la valutazione
            metrics = model.evaluate(per_device_eval_batch_size=batch_size)
            results[model_name] = metrics
            logging.info(f"Model {model_name} - Metrics: {metrics}")
        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {e}")

    # Salva i risultati in un file JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Script per valutare più modelli e generare un file JSON con i risultati')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help="Lista dei nomi dei modelli da valutare (es. 'bart-base-imdb' 'bert-base-uncased-imdb')")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Numero massimo di campioni da usare per la valutazione (utile per debug)")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size per la valutazione")
    parser.add_argument('--output_file', type=str, default='results.json',
                        help="File JSON dove salvare i risultati")
    args = parser.parse_args()

    evaluate_models(args.models, args.max_samples, args.batch_size, args.output_file)

if __name__ == '__main__':
    main()