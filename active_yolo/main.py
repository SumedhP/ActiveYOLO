import argparse

from active_learning.selection import suggest_active_learning_images
from embed import generate_embeddings
from generate_dataset import generate_dataset
from label_images import LabelingTool
from train import train_backbone, train_model
from qualitive_video import qualitive_video
from config import AppConfig


def run_gui():
    """Launches the labeling GUI."""
    app = LabelingTool()
    app.run()


def run_train(ssl=False):
    """Runs model training."""
    if ssl:
        print("Starting Self-Supervised Backbone Training...")
        train_backbone()
    else:
        print("Starting YOLO Training...")
        train_model()


def run_dataset():
    """Generates the dataset."""
    print("Generating dataset from labels...")
    generate_dataset()


def run_embed():
    """Generates embeddings."""
    print("Generating embeddings...")
    generate_embeddings()


def run_suggest():
    """Runs active learning suggestion."""
    print("Selecting images for next annotation batch...")
    suggest_active_learning_images()


def run_video(model_path: str):
    """Generates qualitative video."""
    print("Generating qualitative video...")
    app_config = AppConfig.load_app_config()

    qualitive_video(
        model_path=model_path,
        image_folder=app_config.validation_images_path,
        output_path=app_config.output_path,
    )

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

    # Subcommand: video
    video_parser = subparsers.add_parser("video", help="Generate qualitative video")
    video_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained model"
    )

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
    elif args.command == "video":
        run_video(model_path=args.model_path)


if __name__ == "__main__":
    main()
