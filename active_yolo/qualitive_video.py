from config import AppConfig
from ultralytics import YOLO
import argparse
import cv2

IMAGE_SIZE = 640
CONF = 0.5
HALF = False


def qualitive_video(model_path: str, image_folder: str, output_path: str):
    """Generates a qualitative video using the trained YOLO model."""
    model = YOLO(model_path)

    results = model.predict(
        source=image_folder, conf=CONF, imgsz=IMAGE_SIZE, half=HALF, stream=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = None

    for result in results:
        annotated_frame = result.plot()

        if video_writer is None:
            height, width, _ = annotated_frame.shape
            video_writer = cv2.VideoWriter(
                f"{output_path}/qualitative_video.mp4", fourcc, 30.0, (width, height)
            )

        video_writer.write(annotated_frame)

    if video_writer:
        video_writer.release()

    print(f"Qualitative video saved to {output_path}/qualitative_video.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Qualitative Video using YOLO Model"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained YOLO model"
    )

    args = parser.parse_args()

    app_config = AppConfig.load_app_config()

    qualitive_video(
        model_path=args.model,
        image_folder=app_config.validation_images_path,
        output_path=app_config.output_path,
    )
