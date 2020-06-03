from pathlib import Path

from PIL import Image, UnidentifiedImageError
from detr import DETR, draw_boxes
import typer
import typer


def main(
    images_path: Path = Path("test-images"),
    backbone: str = "detr_resnet50",
    threshold: float = 0.7,
    device: str = "cpu",
):

    model = DETR(backbone=backbone, threshold=threshold, device=device)

    for image_path in images_path.iterdir():
        try:
            with Image.open(image_path) as image:

                scores, boxes = model.predict(image)

                print("N objects:", len(scores))
                draw_boxes(image, scores, boxes)
        except UnidentifiedImageError:
            pass


if __name__ == "__main__":
    typer.run(main)
