import json
import os
import argparse
import logging
from src.evaluate import evaluate_model
from src.data_preprocessing import load_imdb_dataset, create_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_models(model_dirs, test_data, output_file):
    """
    Evaluate multiple models and save results to a JSON file.
    """
    results = {}
    
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        logging.info(f"Evaluating model: {model_name}")
        
        try:
            metrics = evaluate_model(model_dir, test_data)
            results[model_name] = metrics
            logging.info(f"Model {model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {e}")
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate results JSON for model evaluation')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--output_file', type=str, default='results.json',
                        help='Output JSON file to save results')
    args = parser.parse_args()
    
    # Load dataset
    logging.info("Loading dataset...")
    dataset = load_imdb_dataset()
    _, _, test_data = create_splits(dataset)
    
    # Get model directories
    model_dirs = []
    for model_name in os.listdir(args.models_dir):
        model_path = os.path.join(args.models_dir, model_name)
        if os.path.isdir(model_path):
            model_dirs.append(model_path)
    
    if not model_dirs:
        logging.error(f"No model directories found in {args.models_dir}")
        return
    
    # Evaluate models and generate results
    evaluate_models(model_dirs, test_data, args.output_file)

if __name__ == '__main__':
    main()