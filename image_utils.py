from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


class ImageInputError(ValueError):
    """An uploaded file is not a supported, decodable image."""


def read_uploaded_image(
    upload: FileStorage,
    upload_folder: Path,
    allowed_extensions: set[str],
    max_pixels: int,
) -> tuple[np.ndarray, Path]:
    original_name = secure_filename(upload.filename or "")
    if not original_name or "." not in original_name:
        raise ImageInputError("Choose a JPG, JPEG, PNG, or BMP image.")

    extension = original_name.rsplit(".", 1)[1].lower()
    if extension not in allowed_extensions:
        raise ImageInputError("Unsupported file type. Use JPG, JPEG, PNG, or BMP.")

    file_bytes = upload.read()
    if not file_bytes:
        raise ImageInputError("The uploaded file is empty.")

    image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageInputError("The file is not a valid or readable image.")
    height, width = image.shape[:2]
    if height * width > max_pixels:
        raise ImageInputError(
            f"Image is too large to process. Maximum dimensions are {max_pixels:,} total pixels."
        )

    upload_folder.mkdir(parents=True, exist_ok=True)
    image_path = upload_folder / f"{uuid4().hex}.{extension}"
    if not cv2.imwrite(str(image_path), image):
        raise OSError("Could not save the uploaded image.")
    return image, image_path


def save_image(image: np.ndarray, output_folder: Path, prefix: str) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / f"{prefix}_{uuid4().hex}.png"
    if not cv2.imwrite(str(path), image):
        raise OSError("Could not save the processed image.")
    return path


def static_file_url(path: Path, static_folder: Path) -> str:
    return path.relative_to(static_folder).as_posix()


def image_brightness(image: np.ndarray) -> float:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(float(np.mean(grayscale)), 2)
