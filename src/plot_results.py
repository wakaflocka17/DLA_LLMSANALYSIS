import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_results(results_file):
    """
    Load results from a JSON file.
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
    Plot accuracy and F1 scores for different models.
    """
    if not results:
        logging.error("No results to plot")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and metrics
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    f1_scores = [results[model]['f1'] for model in model_names]
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy
    bars1 = ax1.bar(model_names, accuracies, color='skyblue')
    ax1.set_title('Accuracy by Model')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    # Plot F1 scores
    bars2 = ax2.bar(model_names, f1_scores, color='lightgreen')
    ax2.set_title('F1 Score by Model')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {os.path.join(output_dir, 'model_comparison.png')}")
    
    # Create a combined metrics plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='lightgreen')
    
    ax.set_title('Model Performance Comparison')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Combined plot saved to {os.path.join(output_dir, 'combined_metrics.png')}")
    
    # Show plots if running in interactive mode
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot model evaluation results')
    parser.add_argument('--results_file', type=str, default='results.json', 
                        help='JSON file containing model evaluation results')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots')
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    
    # Plot metrics
    plot_metrics(results, args.output_dir)

if __name__ == '__main__':
    main()