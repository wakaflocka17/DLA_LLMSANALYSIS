import argparse
import logging
from model_factory import get_model

# Configurazione del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Evaluation script per i modelli.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Nome del modello da valutare (es. 'bart-base-imdb', 'bert-base-uncased-imdb', ecc.).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per dispositivo durante la valutazione.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Numero massimo di campioni da usare per la valutazione (utile per debug).")
    parser.add_argument("--output_json_path", type=str, default=None,
                        help="Percorso del file JSON in cui salvare i risultati della valutazione.")
    args = parser.parse_args()

    logger.info(f"Inizializzo l'evaluation per il modello: {args.model_name}")
    model = get_model(args.model_name)
    model.prepare_datasets(max_samples=args.max_samples)
    results = model.evaluate(
        per_device_eval_batch_size=args.batch_size,
        output_json_path=args.output_json_path
    )
    logger.info(f"Risultati evaluation: {results}")

if __name__ == "__main__":
    main()