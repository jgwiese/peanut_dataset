import pytest
import numpy as np
from processing import process


@pytest.fixture
def image():
    return np.random.random((512, 512, 3))


def test_process(image):
    image_out = process(image)
    assert image_out.shape == image.shape
