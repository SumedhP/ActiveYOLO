import argparse

from ultralytics import YOLO


def convert_to_percentage(value: str) -> float:
    return round(float(value) * 100, 3)


def get_checkpoint_info(ckpt_path: str) -> None:
    model = YOLO(ckpt_path)

    train_metrics = model.ckpt.get("train_metrics", {})
    print(f"\tEpoch: {model.ckpt.get('epoch', 'N/A')}")
    print(f"\tBest fitness: {model.ckpt.get('best_fitness', 'N/A')}")
    mAP50 = train_metrics.get("metrics/mAP50(B)", "N/A")
    if mAP50 != "N/A":
        mAP50 = convert_to_percentage(mAP50)
    mAP50_95 = train_metrics.get("metrics/mAP50-95(B)", "N/A")
    if mAP50_95 != "N/A":
        mAP50_95 = convert_to_percentage(mAP50_95)
    precision = train_metrics.get("metrics/precision(B)", "N/A")
    if precision != "N/A":
        precision = convert_to_percentage(precision)
    recall = train_metrics.get("metrics/recall(B)", "N/A")
    if recall != "N/A":
        recall = convert_to_percentage(recall)
    print(f"\tmAP50: {mAP50}%")
    print(f"\tmAP50-95: {mAP50_95}%")
    print(f"\tPrecision: {precision}%")
    print(f"\tRecall: {recall}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get YOLO checkpoint information.")
    parser.add_argument(
        "run_path",
        type=str,
        help="Path to run folder containing weights and checkpoints.",
    )
    args = parser.parse_args()

    run_path = args.run_path.rstrip("/")
    best_model = f"{run_path}/weights/best.pt"
    last_model = f"{run_path}/weights/last.pt"

    print("Best Model Checkpoint Info:")
    get_checkpoint_info(best_model)
    print("\nLast Model Checkpoint Info:")
    get_checkpoint_info(last_model)
