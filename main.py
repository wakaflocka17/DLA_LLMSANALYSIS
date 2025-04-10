import argparse
from src.data_preprocessing import load_imdb_dataset, create_splits
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="models/bert")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    set_seed(42)

    # We load the dataset
    dataset = load_imdb_dataset()
    train_data, val_data, test_data = create_splits(dataset)

    # We train the model
    train_model(
        model_name_or_path=args.model_name,
        train_dataset=train_data,
        val_dataset=val_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    # We evaluate the model
    metrics = evaluate_model(args.output_dir, test_data)
    print("Final Metrics:", metrics)

if __name__ == "__main__":
    main()