from io import BytesIO

import cv2
import numpy as np
import pytest
from werkzeug.datastructures import FileStorage

from image_utils import ImageInputError, image_brightness, read_uploaded_image


def make_upload(image, filename="sample.png"):
    success, encoded = cv2.imencode(".png", image)
    assert success
    return FileStorage(stream=BytesIO(encoded.tobytes()), filename=filename)


def test_upload_is_decoded_and_saved_with_generated_name(tmp_path):
    image = np.full((32, 48, 3), 80, dtype=np.uint8)
    decoded, path = read_uploaded_image(
        make_upload(image, "../../my image.PNG"),
        tmp_path,
        {"jpg", "jpeg", "png", "bmp"},
        100_000,
    )
    assert decoded.shape == image.shape
    assert path.parent == tmp_path
    assert path.name.endswith(".png")
    assert path.exists()
    assert image_brightness(decoded) == 80


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("bad.gif", b"not an image", "Unsupported file type"),
        ("", b"not an image", "Choose a JPG"),
        ("broken.jpg", b"not an image", "not a valid"),
    ],
)
def test_rejects_invalid_uploads(tmp_path, filename, content, message):
    upload = FileStorage(stream=BytesIO(content), filename=filename)
    with pytest.raises(ImageInputError, match=message):
        read_uploaded_image(upload, tmp_path, {"jpg", "jpeg", "png", "bmp"}, 100_000)


def test_rejects_image_over_pixel_limit(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ImageInputError, match="too large"):
        read_uploaded_image(make_upload(image), tmp_path, {"png"}, 100)
