import argparse
import logging
from model_factory import get_model
from src.utils import TqdmLoggingCallback

# Configurazione del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Training script per i modelli.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Nome del modello da addestrare (es. 'bart-base-imdb', 'bert-base-uncased-imdb', ecc.).")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory di output dove salvare il modello addestrato.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Numero di epoche di training.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per dispositivo.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Numero massimo di campioni da usare per il training (utile per debug).")
    args = parser.parse_args()

    logger.info(f"Inizializzo il training per il modello: {args.model_name}")
    model = get_model(args.model_name)
    model.prepare_datasets(max_samples=args.max_samples)
    
    # Create the callback with desired update frequency
    tqdm_callback = TqdmLoggingCallback(update_every=10)
    
    # Pass the callback to the train method
    model.train(output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                callbacks=[tqdm_callback])  # Add the callback here

if __name__ == "__main__":
    main()