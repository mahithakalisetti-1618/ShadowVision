import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity


def extract_hog_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if min(grayscale.shape) < 64:
        grayscale = cv2.resize(grayscale, (64, 64), interpolation=cv2.INTER_CUBIC)

    features, visualization = hog(
        grayscale,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    visualization = cv2.normalize(
        visualization, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return features.astype(np.float32), cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)


def hog_cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("HOG descriptors must have matching dimensions.")
    return round(float(cosine_similarity(first.reshape(1, -1), second.reshape(1, -1))[0, 0]), 4)
