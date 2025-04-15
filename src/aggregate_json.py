import os
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_json_files(input_dir, output_file="aggregate_results.json"):
    """
    Legge tutti i file .json in input_dir e li aggrega in un unico file JSON.
    """
    aggregated = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            logging.info(f"Found JSON file: {file_path}")

            with open(file_path, "r") as f:
                data = json.load(f)

            # Estrazione nome modello e scenario dal nome file
            base_name = filename.replace(".json", "")
            parts = base_name.split("_")
            if len(parts) == 2:
                model_name, scenario = parts
            else:
                model_name = base_name
                scenario = "unknown"

            if model_name not in aggregated:
                aggregated[model_name] = {}
            aggregated[model_name][scenario] = data

    # Salvataggio
    output_path = os.path.join(input_dir, output_file)
    with open(output_path, "w") as out_f:
        json.dump(aggregated, out_f, indent=4)

    logging.info(f"Aggregated results saved to {output_path}")
    return aggregated

def main():
    parser = argparse.ArgumentParser(description="Aggrega file JSON contenenti metriche dei modelli.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory contenente i file .json da aggregare")
    parser.add_argument('--output_file', type=str, default="aggregate_results.json", help="Nome del file di output aggregato")
    args = parser.parse_args()

    aggregate_json_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()