import argparse
import logging
import os
import torch # Added torch
from accelerate import Accelerator # Added Accelerator

# Update imports to use the src prefix
from src.model_factory import get_model
from src.model_configs import MODEL_CONFIGS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set CUDA_VISIBLE_DEVICES by default if not already set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

def main():
    parser = argparse.ArgumentParser(description="Script principale per training ed evaluation dei modelli.")
    parser.add_argument("--model_config_key", type=str, required=True,
                        help="Chiave del modello in model_configs (es. 'bart_base', 'bert_base_uncased', 'gpt_neo_2_7b', 'ensemble_majority_voting').")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="eval", # Changed default to eval for optimization focus
                        help="Modalità: train per addestramento, eval per valutazione.")
    parser.add_argument("--eval_type", type=str, choices=["pretrained", "fine_tuned"], default="fine_tuned",
                        help="Tipo di evaluation: 'pretrained' se si vuole valutare il modello pre-addestrato, 'fine_tuned' per il modello fine-tunato.")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory radice in cui salvare i modelli (default: 'models').")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Numero di epoche di training.")
    parser.add_argument("--train_batch_size", type=int, default=None,
                        help="Batch size per dispositivo durante il training.")
    parser.add_argument("--eval_batch_size", type=int, default=None, # Will be overridden by config or new default
                        help="Batch size per dispositivo durante l'evaluation.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Numero massimo di campioni da usare (utile per debug).")
    parser.add_argument("--output_json_path", type=str, default=None,
                    help="Percorso del file JSON in cui salvare i risultati di evaluation.")
    
    # New arguments for optimization
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda' or 'cpu'. Accelerator will manage this primarily.")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable Automatic Mixed Precision (AMP) using torch.float16.")
    parser.add_argument("--use_bettertransformer", action="store_true",
                        help="Enable BetterTransformer for compatible models.")
    parser.add_argument("--use_onnxruntime", action="store_true",
                        help="Enable ONNX Runtime for inference if ONNX model is available.")
    # Batch size specifically for this optimization task, defaulting to 256 as requested
    parser.add_argument("--optimized_eval_batch_size", type=int, default=256,
                        help="Evaluation batch size for optimized run (e.g., 64, 128, 256).")


    args = parser.parse_args()

    # Initialize Accelerator
    mixed_precision_config = "fp16" if args.use_amp else None
    accelerator = Accelerator(mixed_precision=mixed_precision_config)
    logger.info(f"Accelerator initialized. Device: {accelerator.device}, Mixed Precision: {accelerator.mixed_precision}")

    # Override device if specified, but Accelerator's device is preferred
    if args.device:
        logger.warning(f"--device argument ({args.device}) provided, but Accelerator will manage device placement ({accelerator.device}).")
    
    effective_eval_batch_size = args.optimized_eval_batch_size

    # Se esiste una configurazione per il modello, usiamo i valori di default se non specificati
    if args.model_config_key in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model_config_key]
        if args.epochs is None:
            args.epochs = config.get("epochs")
        if args.train_batch_size is None:
            args.train_batch_size = config.get("train_batch_size", config.get("batch_size"))
        
        # For eval_batch_size, prioritize optimized_eval_batch_size, then command-line, then config
        if args.eval_batch_size is not None: # if --eval_batch_size was explicitly passed
             effective_eval_batch_size = args.eval_batch_size
        elif "eval_batch_size" in config: # if not passed, try from config
             effective_eval_batch_size = config.get("eval_batch_size")
        # Otherwise, effective_eval_batch_size remains args.optimized_eval_batch_size (default 256)

        logger.info(f"Usando la configurazione di default: {config}")
        logger.info(f"Effective evaluation batch size: {effective_eval_batch_size}")
    else:
        logger.warning("Chiave modello non trovata in model_configs. Verranno usati solo i parametri da riga di comando.")
        # Use optimized_eval_batch_size if no config and no specific --eval_batch_size
        if args.eval_batch_size is not None:
            effective_eval_batch_size = args.eval_batch_size


    # Componiamo la directory finale di output (es. 'models/bert_base_uncased')
    output_dir = os.path.join(args.output_dir, args.model_config_key)
    logger.info(f"I modelli verranno salvati in: {output_dir}")

    # Creiamo l'istanza del modello tramite la factory, passing accelerator and optimization flags
    model_init_kwargs = {
        "accelerator": accelerator,
        "use_amp": args.use_amp,
        "use_bettertransformer": args.use_bettertransformer,
        "use_onnxruntime": args.use_onnxruntime,
    }
    # For ensemble, it might take the device from accelerator directly.
    # For individual models, they might need 'device' if not using accelerator fully internally.
    # model_factory will pass these to the specific model's __init__
    
    model = get_model(args.model_config_key, **model_init_kwargs)


    # Prepara i dataset
    # This method is part of the model class structure, ensure it's compatible
    # or adjust how datasets are handled with Accelerator
    model.prepare_datasets(max_samples=args.max_samples)

    if args.mode == "train":
        logger.info(f"Modalità TRAIN: avvio del training per il modello {args.model_config_key}.")
        # Training logic would also need to be adapted for Accelerator
        # model.train(...) would need to use accelerator.prepare for model, optimizer, dataloaders
        # and accelerator.backward() for loss.
        # This is out of scope for the current optimization task which focuses on evaluation.
        logger.warning("Training mode with full Accelerator integration is not implemented in this patch.")
        model.train(output_dir=output_dir,
                    num_train_epochs=args.epochs,
                    per_device_train_batch_size=args.train_batch_size) # Original train call
    else: # eval mode
        logger.info(f"Modalità EVAL: avvio dell'evaluation per il modello {args.model_config_key}.")
        
        eval_kwargs = {
            "per_device_eval_batch_size": effective_eval_batch_size,
            "output_json_path": args.output_json_path
            # num_dataloader_workers will be handled inside evaluate method
        }

        if args.eval_type == "fine_tuned":
            result = model.evaluate(**eval_kwargs)
        else: # pretrained
            # Assuming evaluate_pretrained also needs similar optimization parameters
            # This might require changes in evaluate_pretrained if it exists and is used.
            # For now, we assume 'evaluate' is the primary target for optimization.
            if hasattr(model, 'evaluate_pretrained'):
                result = model.evaluate_pretrained(**eval_kwargs)
            else:
                logger.warning(f"evaluate_pretrained not implemented for {args.model_config_key}, falling back to evaluate().")
                result = model.evaluate(**eval_kwargs)
        logger.info(f"Risultati evaluation: {result}")

        # Save the ensemble model if it's the ensemble_majority_voting model
        if args.model_config_key == 'ensemble_majority_voting' and hasattr(model, 'save'):
            if 'repo' in MODEL_CONFIGS['ensemble_majority_voting']:
                save_path = MODEL_CONFIGS['ensemble_majority_voting']['repo']
                logger.info(f"Attempting to save ensemble model to: {save_path}")
                model.save(save_path) # Call the new save method
                # The save method itself logs success/failure only on the main process
                if accelerator.is_local_main_process:
                    logger.info(f"Ensemble model saving process initiated. Check logs for details. Target path: {save_path}")
            else:
                logger.error(f"Cannot save ensemble: 'repo' path not defined in MODEL_CONFIGS for '{args.model_config_key}'.")


if __name__ == "__main__":
    main()