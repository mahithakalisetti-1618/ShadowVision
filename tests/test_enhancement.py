import cv2
import numpy as np
import pytest

from enhancement import (
    adaptive_threshold,
    apply_spatial_filter,
    detect_edges,
    enhance_image,
    equalize_histogram,
    gamma_correction,
)


@pytest.fixture
def sample_image():
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[20:76, 30:100] = (45, 70, 95)
    image[38:58, 49:82] = (120, 145, 170)
    return image


def test_gamma_correction_lifts_dark_tones(sample_image):
    corrected = gamma_correction(sample_image, 1.8)
    assert corrected.shape == sample_image.shape
    assert corrected.dtype == np.uint8
    assert corrected.mean() > sample_image.mean()


def test_gamma_rejects_out_of_range_value(sample_image):
    with pytest.raises(ValueError, match="between 0.1 and 3.0"):
        gamma_correction(sample_image, 0)


def test_histogram_equalization_preserves_color_shape(sample_image):
    equalized = equalize_histogram(sample_image)
    assert equalized.shape == sample_image.shape
    assert equalized.dtype == np.uint8
    assert not np.array_equal(equalized, sample_image)


@pytest.mark.parametrize("filter_name", ["none", "gaussian", "median", "bilateral", "sharpen"])
def test_spatial_filters_return_valid_images(sample_image, filter_name):
    filtered = apply_spatial_filter(sample_image, filter_name)
    assert filtered.shape == sample_image.shape
    assert filtered.dtype == np.uint8


def test_edge_detectors_and_adaptive_threshold_produce_images(sample_image):
    assert detect_edges(sample_image, "canny").shape == sample_image.shape[:2]
    assert detect_edges(sample_image, "sobel").shape == sample_image.shape[:2]
    assert detect_edges(sample_image, "none") is None
    thresholded = adaptive_threshold(sample_image)
    assert thresholded.shape == sample_image.shape[:2]
    assert set(np.unique(thresholded)).issubset({0, 255})


def test_full_enhancement_pipeline(sample_image):
    enhanced = enhance_image(sample_image, gamma=1.5, equalize=True, filter_name="sharpen")
    assert enhanced.shape == sample_image.shape
    assert cv2.absdiff(enhanced, sample_image).any()
