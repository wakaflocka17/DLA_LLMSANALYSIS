import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_results(results_file):
    """
    Carica i risultati da un file JSON.
    """
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Results file not found: {results_file}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {results_file}")
        return {}

def plot_metrics(results, output_dir):
    """
    Si aspetta un dizionario con questa struttura:
    {
       "bart": {
          "finetuned": { "accuracy": 0.9,  "f1": 0.87, ... },
          "pretrained": { "accuracy": 0.75, "f1": 0.72, ... }
       },
       "gpt-neo": {
          "finetuned":   { "accuracy": 0.88, "f1": 0.84, ... },
          "pretrained":  { "accuracy": 0.71, "f1": 0.70, ... }
       }
    }
    """
    if not results:
        logging.error("No results to plot. Check your JSON structure or input file.")
        return
    
    # Creiamo la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Estraggo i nomi dei modelli
    model_names = list(results.keys())  # Esempio: ["bart", "gpt-neo-2.7b", ...]
    
    # Liste per accuracy finetuned e pretrained
    accuracies_finetuned = []
    accuracies_pretrained = []
    
    for model_name in model_names:
        # Recuperiamo i risultati "finetuned" e "pretrained" (se presenti)
        ft_metrics = results[model_name].get("finetuned", {})
        pt_metrics = results[model_name].get("pretrained", {})
        
        accuracies_finetuned.append(ft_metrics.get("accuracy", 0.0))
        accuracies_pretrained.append(pt_metrics.get("accuracy", 0.0))
    
    # Creiamo un grafico a barre per confrontare l'Accuracy (finetuned vs pretrained)
    x = np.arange(len(model_names))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars_ft = ax.bar(x - width/2, accuracies_finetuned, width, label='Finetuned')
    bars_pt = ax.bar(x + width/2, accuracies_pretrained, width, label='Pretrained')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha='right')  # Rotazione se i nomi sono lunghi
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparison of Accuracy: Finetuned vs Pretrained")
    ax.legend()
    
    # Aggiunge le label con il valore sulle barre
    for bar in bars_ft + bars_pt:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    logging.info(f"Plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot model evaluation results')
    parser.add_argument('--results_file', type=str, default='results_aggregati.json', 
                        help='JSON file containing aggregated model evaluation results')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots')
    args = parser.parse_args()
    
    # Carichiamo il JSON aggregato
    results = load_results(args.results_file)
    
    # Generiamo il grafico
    plot_metrics(results, args.output_dir)

if __name__ == '__main__':
    main()