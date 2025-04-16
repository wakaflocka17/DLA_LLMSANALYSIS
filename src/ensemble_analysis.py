import json
import argparse
import os
import logging
import matplotlib.pyplot as plt
from collections import Counter

logging.basicConfig(level=logging.INFO)

def analyze_divergences(ensemble_json, output_dir="plots"):
    detailed = ensemble_json.get("detailed_predictions", [])
    if not detailed:
        logging.warning("Nessuna predizione dettagliata trovata.")
        return

    disagreements = []
    all_model_names = [k for k in detailed[0].keys() if k not in ["true_label", "voted"]]

    for i, pred in enumerate(detailed):
        votes = [pred[m] for m in all_model_names]
        if len(set(votes)) > 1:
            disagreements.append({
                "index": i,
                "true_label": pred["true_label"],
                "votes": {m: pred[m] for m in all_model_names},
                "voted": pred["voted"]
            })

    logging.info(f"Totale esempi con disaccordo: {len(disagreements)} su {len(detailed)}")

    disagreement_counter = Counter()
    for d in disagreements:
        for model, prediction in d["votes"].items():
            if prediction != d["voted"]:
                disagreement_counter[model] += 1

    print("\n🧮 Disaccordi per modello:")
    for model, count in disagreement_counter.items():
        print(f"- {model}: {count} volte in disaccordo col voto finale")

    print("\n📌 Esempi in disaccordo (prime 5):")
    for d in disagreements[:5]:
        print(f"Index {d['index']} | True: {d['true_label']} | Votes: {d['votes']} → Voted: {d['voted']}")

    # ==== GRAFICO ====
    if disagreement_counter:
        os.makedirs(output_dir, exist_ok=True)
        model_names = list(disagreement_counter.keys())
        counts = list(disagreement_counter.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(model_names, counts)

        ax.set_ylabel("Numero di disaccordi col voto finale")
        ax.set_title("Divergenze tra modelli nell'ensemble")
        plt.xticks(rotation=15)

        # Aggiungi label sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height}', ha='center', va='bottom')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "ensemble_disagreements.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()
        logging.info(f"Grafico salvato in: {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_file", type=str, required=True, help="Path al file JSON dell'ensemble")
    parser.add_argument("--output_dir", type=str, default="plots", help="Cartella in cui salvare il grafico")
    args = parser.parse_args()

    if not os.path.exists(args.ensemble_file):
        logging.error(f"File non trovato: {args.ensemble_file}")
        return

    with open(args.ensemble_file, "r") as f:
        ensemble_data = json.load(f)

    analyze_divergences(ensemble_data, output_dir=args.output_dir)

if __name__ == "__main__":
    main()