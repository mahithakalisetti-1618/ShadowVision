from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_histogram_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    output_folder: Path,
    file_name: str,
) -> Path:
    output_folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    colors = ((0, "blue", "Blue"), (1, "green", "Green"), (2, "red", "Red"))
    for axis, image, title in (
        (axes[0], original, "Original"),
        (axes[1], enhanced, "Enhanced"),
    ):
        for channel, color, label in colors:
            histogram = cv2.calcHist([image], [channel], None, [256], [0, 256])
            axis.plot(histogram, color=color, linewidth=1.1, label=label)
        axis.set(title=title, xlim=(0, 255), xlabel="Intensity", ylabel="Pixel count")
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, ncol=3, fontsize=8)

    figure.suptitle("Color intensity distribution")
    path = output_folder / file_name
    figure.savefig(path, dpi=150, facecolor="#ffffff")
    plt.close(figure)
    return path


def histogram_data(image: np.ndarray) -> dict[str, list[int]]:
    channels = ("blue", "green", "red")
    return {
        name: cv2.calcHist([image], [index], None, [256], [0, 256]).flatten().astype(int).tolist()
        for index, name in enumerate(channels)
    }
