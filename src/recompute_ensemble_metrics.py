import json
import argparse
import sys

def compute_metrics_from_detailed_predictions(detailed_predictions):
    """
    Computes Accuracy, Precision, Recall, and F1 based on 'voted' vs 'true_label'
    from a list of detailed prediction objects.
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    tn = 0  # True Negatives

    if not detailed_predictions:
        # If there are no predictions, all metrics are 0.
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for pred_item in detailed_predictions:
        voted = pred_item.get('voted')
        true_label = pred_item.get('true_label')

        if voted == 1 and true_label == 1:
            tp += 1
        elif voted == 1 and true_label == 0:
            fp += 1
        elif voted == 0 and true_label == 1:
            fn += 1
        elif voted == 0 and true_label == 0:
            tn += 1
        # Assuming 'voted' and 'true_label' are always 0 or 1 as per binary classification.

    total_samples = tp + tn + fp + fn
    
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser(
        description="Compute classification metrics from an ensemble JSON file's detailed predictions."
    )
    parser.add_argument(
        "ensemble_json_path", 
        type=str, 
        help="Path to the input ensemble JSON file."
    )
    args = parser.parse_args()

    try:
        with open(args.ensemble_json_path, 'r') as f:
            ensemble_data_full = json.load(f)
    except FileNotFoundError:
        error_message = {"error": f"File not found at {args.ensemble_json_path}"}
        print(json.dumps(error_message), file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        error_message = {"error": f"Invalid JSON in file {args.ensemble_json_path}"}
        print(json.dumps(error_message), file=sys.stderr)
        sys.exit(1)

    detailed_predictions_list = ensemble_data_full.get("detailed_predictions_main_process")

    if detailed_predictions_list is None:
        # If the key is missing, the problem implies we should still compute.
        # The compute_metrics_from_detailed_predictions function handles an empty or None list.
        # However, it's better to inform the user if the expected data structure is not present.
        error_message = {"error": f"'detailed_predictions_main_process' key not found in {args.ensemble_json_path}. Cannot compute metrics as specified."}
        print(json.dumps(error_message), file=sys.stderr)
        # Outputting a JSON with all zeros if the critical part is missing,
        # or exiting might be options. Let's output zeros as the function would.
        computed_metrics = compute_metrics_from_detailed_predictions(None)

    else:
        computed_metrics = compute_metrics_from_detailed_predictions(detailed_predictions_list)
    
    # Print ONLY the JSON object to stdout, with indent 2 to match the example format
    print(json.dumps(computed_metrics, indent=2))

if __name__ == "__main__":
    main()