import cv2
import numpy as np


FILTERS = {"none", "gaussian", "median", "bilateral", "sharpen"}
EDGE_METHODS = {"none", "canny", "sobel"}


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    if not 0.1 <= gamma <= 3.0:
        raise ValueError("Gamma must be between 0.1 and 3.0.")
    lookup = np.array(
        [((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lookup)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_spatial_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    if filter_name not in FILTERS:
        raise ValueError("Choose a supported spatial filter.")
    if filter_name == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if filter_name == "median":
        return cv2.medianBlur(image, 5)
    if filter_name == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    if filter_name == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    return image.copy()


def detect_edges(image: np.ndarray, method: str = "canny") -> np.ndarray | None:
    if method not in EDGE_METHODS:
        raise ValueError("Choose Canny, Sobel, or no edge detection.")
    if method == "none":
        return None

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "canny":
        return cv2.Canny(grayscale, 80, 160)

    horizontal = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(horizontal, vertical)
    return cv2.convertScaleAbs(magnitude)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if min(grayscale.shape) < 3:
        raise ValueError("Image must be at least 3 pixels wide and high for adaptive thresholding.")
    block_size = min(31, min(grayscale.shape) if min(grayscale.shape) % 2 else min(grayscale.shape) - 1)
    block_size = max(3, block_size)
    return cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        2,
    )


def enhance_image(
    image: np.ndarray,
    gamma: float = 1.0,
    equalize: bool = False,
    filter_name: str = "none",
) -> np.ndarray:
    enhanced = gamma_correction(image, gamma)
    if equalize:
        enhanced = equalize_histogram(enhanced)
    return apply_spatial_filter(enhanced, filter_name)
