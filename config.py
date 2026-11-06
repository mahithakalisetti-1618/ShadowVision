import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SHADOWVISION_SECRET_KEY", "shadowvision-local-development")
    MAX_CONTENT_LENGTH = 12 * 1024 * 1024
    UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"
    MODEL_PATH = Path(
        os.environ.get("SHADOWVISION_MODEL", str(BASE_DIR / "models" / "yolo11n-seg.pt"))
    ).expanduser()
    MAX_IMAGE_PIXELS = 40_000_000
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
