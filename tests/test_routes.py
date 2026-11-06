from io import BytesIO

import cv2
import numpy as np
import pytest

from app import create_app


@pytest.fixture
def client(tmp_path):
    static_folder = tmp_path / "static"
    application = create_app(
        {
            "TESTING": True,
            "STATIC_FOLDER": str(static_folder),
            "UPLOAD_FOLDER": static_folder / "uploads",
            "OUTPUT_FOLDER": static_folder / "outputs",
            "MODEL_PATH": tmp_path / "missing-model.pt",
            "MAX_CONTENT_LENGTH": 1024 * 1024,
        }
    )
    return application.test_client()


def encoded_image():
    image = np.full((96, 128, 3), (40, 70, 100), dtype=np.uint8)
    image[20:70, 30:95] = (140, 160, 180)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


@pytest.mark.parametrize(
    "path",
    ["/", "/analysis", "/detection", "/segmentation", "/analytics", "/about"],
)
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"shadowvision" in response.data.lower()


def test_unknown_page_renders_custom_not_found(client):
    response = client.get("/not-a-page")
    assert response.status_code == 404
    assert b"out of frame" in response.data.lower()


def test_enhancement_api_returns_real_saved_outputs(client):
    response = client.post(
        "/api/enhance",
        data={
            "image": (BytesIO(encoded_image()), "dim scene.png"),
            "gamma": "1.5",
            "equalize": "true",
            "filter": "gaussian",
            "edge_method": "canny",
        },
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["dimensions"] == {"width": 128, "height": 96}
    assert payload["original_brightness"] != payload["enhanced_brightness"]
    assert payload["hog_feature_count"] > 0
    assert payload["histogram_data"]["blue"][0] >= 0
    for key in ("processed", "edges", "threshold", "hog_visualization", "histogram"):
        file_response = client.get(payload[key])
        assert file_response.status_code == 200
        assert file_response.data


def test_enhancement_api_reports_invalid_gamma(client):
    response = client.post(
        "/api/enhance",
        data={"image": (BytesIO(encoded_image()), "scene.png"), "gamma": "0"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Gamma must" in response.get_json()["error"]


def test_detection_api_reports_missing_model_without_fake_result(client):
    response = client.post(
        "/api/detect",
        data={"image": (BytesIO(encoded_image()), "scene.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    assert "model not found" in response.get_json()["error"]


def test_processing_api_rejects_unsupported_file(client):
    response = client.post(
        "/api/enhance",
        data={"image": (BytesIO(b"file"), "scene.gif")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]
