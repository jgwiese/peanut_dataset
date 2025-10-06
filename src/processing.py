from pathlib import Path
from typing import Optional
import cv2 as cv
import numpy as np

PATH = Path("./data/raw")


def load_image(file: Path):
    image = cv.imread(filename=str(file))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def write_image(file: Path, image):
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(filename=str(file), img=image)
    return image


def custom_perspective_transform(image, points: Optional[np.float32] = None):
    print(image.shape)
    if points is None:
        points = np.float32([[220, 570], [160, 3165], [2790, 605], [2790, 3200]])
    points_target = np.float32([[0, 0], [0, 512], [512, 0], [512, 512]])
    m = cv.getPerspectiveTransform(points, points_target)
    image = cv.warpPerspective(image, m, (512, 512))
    return image


def process(image):
    image = custom_perspective_transform(image)
    return image


def process_image_file(file_input: Path, file_output: Path) -> None:
    image = load_image(file=file_input)
    image = process(image=image)
    write_image(file=file_output, image=image)


if __name__ == "__main__":
    for i, filename in enumerate(sorted(PATH.glob("*.jpg"))[:]):
        print(filename)
        process_image_file(
            file_input=filename,
            file_output=Path(f"data/processed/{i:04d}_image.png"),
        )
