import argparse
import os
from src.data_preprocessing import load_imdb_dataset, create_splits
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import set_seed
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', 
                        default=["bert-base-uncased", "distilbert-base-uncased", "roberta-base"])
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval_only", action="store_true", help="Solo valutazione dei modelli pre-addestrati")
    args = parser.parse_args()

    set_seed(42)

    # We load the dataset
    dataset = load_imdb_dataset()
    train_data, val_data, test_data = create_splits(dataset)
    
    # Configurazione logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("sentiment_analysis.log"),
                            logging.StreamHandler()
                        ])
    
    results = {}
    
    # Per ogni modello nella lista
    for model_name in args.models:
        logging.info(f"Elaborazione del modello: {model_name}")
        model_output_dir = os.path.join(args.output_dir, model_name.split('/')[-1])
        
        # Fase di valutazione pre-addestramento (zero-shot)
        if args.eval_only:
            logging.info(f"Valutazione zero-shot del modello pre-addestrato: {model_name}")
            try:
                zero_shot_metrics = evaluate_model(model_name, test_data, is_pretrained=True)
                logging.info(f"Metriche zero-shot per {model_name}: {zero_shot_metrics}")
                results[f"{model_name}_zero_shot"] = zero_shot_metrics
            except Exception as e:
                logging.error(f"Errore nella valutazione zero-shot di {model_name}: {e}")
        else:
            # Fase di addestramento
            logging.info(f"Addestramento del modello: {model_name}")
            try:
                train_model(
                    model_name_or_path=model_name,
                    train_dataset=train_data,
                    val_dataset=val_data,
                    output_dir=model_output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr
                )
                
                # Fase di valutazione post-addestramento
                logging.info(f"Valutazione del modello addestrato: {model_name}")
                metrics = evaluate_model(model_output_dir, test_data)
                logging.info(f"Metriche per {model_name}: {metrics}")
                results[model_name] = metrics
            except Exception as e:
                logging.error(f"Errore nell'addestramento o valutazione di {model_name}: {e}")
    
    # Stampa dei risultati finali
    logging.info("Riepilogo dei risultati:")
    for model_name, metrics in results.items():
        logging.info(f"{model_name}: {metrics}")
    
    print("\nRisultati finali:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

if __name__ == "__main__":
    main()