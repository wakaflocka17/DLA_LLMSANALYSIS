import argparse
import logging
import os

# Update imports to use the src prefix
from src.model_factory import get_model
from src.model_configs import MODEL_CONFIGS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Script principale per training ed evaluation dei modelli.")
    parser.add_argument("--model_config_key", type=str, required=True,
                        help="Chiave del modello in model_configs (es. 'bart_base', 'bert_base_uncased', 'gpt_neo_2_7b').")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train",
                        help="Modalità: train per addestramento, eval per valutazione.")
    parser.add_argument("--eval_type", type=str, choices=["pretrained", "fine_tuned"], default="fine_tuned",
                        help="Tipo di evaluation: 'pretrained' se si vuole valutare il modello pre-addestrato, 'fine_tuned' per il modello fine-tunato.")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory radice in cui salvare i modelli (default: 'models').")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Numero di epoche di training.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size per dispositivo.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Numero massimo di campioni da usare (utile per debug).")
    args = parser.parse_args()

    # Se esiste una configurazione per il modello, usiamo i valori di default (se non specificati)
    if args.model_config_key in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model_config_key]
        if args.epochs is None:
            args.epochs = config.get("epochs")
        if args.batch_size is None:
            args.batch_size = config.get("batch_size")
        logger.info(f"Usando la configurazione di default: {config}")
    else:
        logger.warning("Chiave modello non trovata in model_configs. Verranno usati solo i parametri da riga di comando.")

    # Componiamo la directory finale di output (ad es. 'models/bert_base_uncased')
    output_dir = os.path.join(args.output_dir, args.model_config_key)
    logger.info(f"I modelli verranno salvati in: {output_dir}")

    # Creiamo l'istanza del modello (usiamo la stessa chiave per la factory).
    model = get_model(args.model_config_key)

    # Prepara i dataset
    model.prepare_datasets(max_samples=args.max_samples)

    if args.mode == "train":
        logger.info(f"Modalità TRAIN: avvio del training per il modello {args.model_config_key}.")
        model.train(output_dir=output_dir,
                    num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.batch_size)
    else:
        logger.info(f"Modalità EVAL: avvio dell'evaluation per il modello {args.model_config_key}.")
        # Differenziamo se si vuole valutare il modello pre-addestrato o quello fine-tunato.
        if args.eval_type == "fine_tuned":
            result = model.evaluate(per_device_eval_batch_size=args.batch_size)
        else:
            # In questo caso non ricarichiamo il modello dalla directory fine-tunata, ma usiamo l'istanza originale.
            result = model.evaluate_pretrained(per_device_eval_batch_size=args.batch_size)
        logger.info(f"Risultati evaluation: {result}")

if __name__ == "__main__":
    main()