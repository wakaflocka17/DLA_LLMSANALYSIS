import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(folder):
    models = {}
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            path = os.path.join(folder, fname)
            with open(path) as f:
                data = json.load(f)
                name = fname.replace("_metrics.json", "")
                models[name] = data
    return models

def plot_metrics(models_dict, title, filename=None):
    metrics = ["accuracy", "precision", "recall", "f1"]
    bar_width = 0.15
    x = np.arange(len(metrics))

    plt.figure(figsize=(10, 6))

    for i, (model_name, model_metrics) in enumerate(models_dict.items()):
        values = [model_metrics.get(k, 0) for k in metrics]
        plt.bar(x + i * bar_width, values, width=bar_width, label=model_name)
        for j, val in enumerate(values):
            plt.text(x[j] + i * bar_width, val + 0.01, f"{val:.3f}", ha='center', fontsize=8)

    plt.xticks(x + bar_width * (len(models_dict) - 1) / 2, [m.capitalize() for m in metrics])
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def main():
    base_dir = "experiments/results"

    # --- VALIDATION ---
    print("\n📊 VALIDATION - Fine-Tuned")
    val_models = load_metrics(os.path.join(base_dir, "validation/finetuned"))
    for name, m in val_models.items():
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")
    plot_metrics(val_models, "Validation Metrics (Fine-Tuned)", "validation_finetuned.png")

    # --- EVALUATION ---
    print("\n📊 EVALUATION - Fine-Tuned")
    eval_ft_models = load_metrics(os.path.join(base_dir, "evaluation/finetuned"))
    for name, m in eval_ft_models.items():
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")
    plot_metrics(eval_ft_models, "Evaluation Metrics (Fine-Tuned)", "evaluation_finetuned.png")

    print("\n📊 EVALUATION - Pretrained")
    eval_pt_models = load_metrics(os.path.join(base_dir, "evaluation/pretrained"))
    for name, m in eval_pt_models.items():
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")
    plot_metrics(eval_pt_models, "Evaluation Metrics (Pretrained)", "evaluation_pretrained.png")

    # --- ENSEMBLE ---
    print("\n📊 EVALUATION - Ensemble")
    ensemble_path = os.path.join(base_dir, "evaluation", "ensemble-majority-voting-imdb_metrics.json")
    if os.path.exists(ensemble_path):
        with open(ensemble_path) as f:
            ensemble_metrics = json.load(f)
        print("\nensemble-majority-voting-imdb:")
        for k, v in ensemble_metrics.items():
            print(f"  {k}: {v}")
        plot_metrics({"Ensemble": ensemble_metrics}, "Evaluation: Ensemble Model", "evaluation_ensemble.png")

    # --- FINE-TUNED vs PRETRAINED ---
    print("\n📊 FINE-TUNED vs PRETRAINED")
    common_models = set(eval_ft_models) & set(eval_pt_models)
    comparison = {}
    for model in common_models:
        comparison[model + "_ft"] = eval_ft_models[model]
        comparison[model + "_pt"] = eval_pt_models[model]
    plot_metrics(comparison, "Fine-Tuned vs Pretrained Models", "comparison_ft_vs_pt.png")

if __name__ == "__main__":
    main()