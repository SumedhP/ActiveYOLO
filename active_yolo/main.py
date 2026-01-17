import argparse


def run_gui():
    """Launches the labeling GUI."""
    from active_yolo.label_images import LabelingTool

    app = LabelingTool()
    app.run()


def run_train(ssl=False):
    """Runs model training."""
    from active_yolo.train import train_backbone, train_model

    if ssl:
        print("Starting Self-Supervised Backbone Training...")
        train_backbone()
    else:
        print("Starting YOLO Training...")
        train_model()


def run_dataset():
    """Generates the dataset."""
    from active_yolo.generate_dataset import generate_dataset

    print("Generating dataset from labels...")
    generate_dataset()


def run_embed():
    """Generates embeddings."""
    from active_yolo.embed import generate_embeddings

    print("Generating embeddings...")
    generate_embeddings()


def run_suggest():
    """Runs active learning suggestion."""
    from active_yolo.active_learning.selection import suggest_active_learning_images

    print("Selecting images for next annotation batch...")
    suggest_active_learning_images()


def main():
    parser = argparse.ArgumentParser(description="ActiveYOLO Pipeline CLI")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Subcommand: label
    subparsers.add_parser("label", help="Launch the GUI labeling tool")

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--ssl", action="store_true", help="Run self-supervised backbone training"
    )

    # Subcommand: dataset
    subparsers.add_parser("dataset", help="Generate YOLO dataset from labels")

    # Subcommand: embed
    subparsers.add_parser("embed", help="Generate embeddings for all images")

    # Subcommand: suggest
    subparsers.add_parser("suggest", help="Suggest next batch of images for labeling")

    args = parser.parse_args()

    if args.command == "label":
        run_gui()
    elif args.command == "train":
        run_train(ssl=args.ssl)
    elif args.command == "dataset":
        run_dataset()
    elif args.command == "embed":
        run_embed()
    elif args.command == "suggest":
        run_suggest()


if __name__ == "__main__":
    main()
