from typing import Any
from numpy._typing import NDArray
import matplotlib.pyplot as plt


def compare_images(image_a: NDArray, image_b: NDArray) -> Any:
    figure = plt.figure()
    ax_a = figure.add_subplot(1, 2, 1)
    ax_b = figure.add_subplot(1, 2, 2)
    ax_a.imshow(image_a)
    ax_b.imshow(image_b)
    return figure
