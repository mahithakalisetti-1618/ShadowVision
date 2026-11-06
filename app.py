from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from analytics import histogram_data, save_histogram_comparison
from config import Config
from detection import ModelUnavailableError, run_segmentation
from enhancement import (
    EDGE_METHODS,
    FILTERS,
    adaptive_threshold,
    detect_edges,
    enhance_image,
)
from feature_extraction import extract_hog_features, hog_cosine_similarity
from image_utils import (
    ImageInputError,
    image_brightness,
    read_uploaded_image,
    save_image,
    static_file_url,
)


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(
        __name__, static_folder=(test_config or {}).get("STATIC_FOLDER", "static")
    )
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)
    static_root = Path(app.static_folder)

    def file_url(path: Path) -> str:
        return url_for("static", filename=static_file_url(path, static_root))

    def upload_image():
        upload = request.files.get("image")
        if upload is None:
            raise ImageInputError("Choose an image to upload.")
        return read_uploaded_image(
            upload,
            Path(app.config["UPLOAD_FOLDER"]),
            set(app.config["ALLOWED_EXTENSIONS"]),
            int(app.config["MAX_IMAGE_PIXELS"]),
        )

    def json_error(message: str, status: int):
        return jsonify({"success": False, "error": message}), status

    @app.errorhandler(413)
    def too_large(_error):
        if request.path.startswith("/api/"):
            return json_error("Image exceeds the 12 MB upload limit.", 413)
        return render_template("error.html", message="That upload exceeds the 12 MB limit."), 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/analysis")
    def analysis_page():
        return render_template("analysis.html")

    @app.get("/detection")
    def detection_page():
        return render_template("detection.html")

    @app.get("/segmentation")
    def segmentation_page():
        return render_template("segmentation.html")

    @app.get("/analytics")
    def analytics_page():
        return render_template("analytics.html")

    @app.get("/about")
    def about_page():
        return render_template("about.html")

    @app.get("/outputs/<path:filename>")
    def download_output(filename: str):
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

    @app.post("/api/enhance")
    def enhance_route():
        try:
            original, original_path = upload_image()
            gamma = float(request.form.get("gamma", "1.0"))
            equalize = request.form.get("equalize", "false").lower() == "true"
            filter_name = request.form.get("filter", "none")
            edge_method = request.form.get("edge_method", "canny")
            if filter_name not in FILTERS or edge_method not in EDGE_METHODS:
                raise ValueError("Choose a supported filter and edge detection method.")

            started = perf_counter()
            processed = enhance_image(original, gamma, equalize, filter_name)
            edges = detect_edges(processed, edge_method)
            thresholded = adaptive_threshold(processed)
            original_hog, _ = extract_hog_features(original)
            processed_hog, hog_visual = extract_hog_features(processed)
            elapsed = round(perf_counter() - started, 4)

            processed_path = save_image(processed, Path(app.config["OUTPUT_FOLDER"]), "enhanced")
            threshold_path = save_image(
                thresholded, Path(app.config["OUTPUT_FOLDER"]), "threshold"
            )
            hog_path = save_image(hog_visual, Path(app.config["OUTPUT_FOLDER"]), "hog")
            edge_path = (
                save_image(edges, Path(app.config["OUTPUT_FOLDER"]), "edges")
                if edges is not None
                else None
            )
            histogram_path = save_histogram_comparison(
                original,
                processed,
                Path(app.config["OUTPUT_FOLDER"]),
                f"histogram_{uuid4().hex}.png",
            )

            return jsonify(
                {
                    "success": True,
                    "original": file_url(original_path),
                    "processed": file_url(processed_path),
                    "edges": file_url(edge_path) if edge_path else None,
                    "threshold": file_url(threshold_path),
                    "hog_visualization": file_url(hog_path),
                    "histogram": file_url(histogram_path),
                    "download": url_for(
                        "download_output", filename=processed_path.name
                    ),
                    "dimensions": {"width": int(original.shape[1]), "height": int(original.shape[0])},
                    "original_brightness": image_brightness(original),
                    "enhanced_brightness": image_brightness(processed),
                    "hog_feature_count": int(processed_hog.size),
                    "hog_similarity": hog_cosine_similarity(original_hog, processed_hog),
                    "processing_time": elapsed,
                    "histogram_data": histogram_data(processed),
                }
            )
        except ImageInputError as error:
            return json_error(str(error), 400)
        except ValueError as error:
            return json_error(str(error), 400)
        except OSError as error:
            app.logger.exception("Could not save enhancement output")
            return json_error(f"Could not save processing output: {error}", 500)
        except Exception as error:
            app.logger.exception("Image enhancement failed")
            return json_error(f"Image enhancement failed: {error}", 500)

    def run_detection_request():
        original, original_path = upload_image()
        confidence = float(request.form.get("confidence", "0.25"))
        results = run_segmentation(
            original, Path(app.config["MODEL_PATH"]), confidence=confidence
        )
        annotated_path = save_image(
            results["annotated"], Path(app.config["OUTPUT_FOLDER"]), "detection"
        )
        return original, original_path, results, annotated_path

    @app.post("/api/detect")
    def detect_route():
        try:
            original, original_path, results, annotated_path = run_detection_request()
            return jsonify(
                {
                    "success": True,
                    "original": file_url(original_path),
                    "result": file_url(annotated_path),
                    "download": url_for("download_output", filename=annotated_path.name),
                    "objects": results["detections"],
                    "object_count": len(results["detections"]),
                    "processing_time": results["processing_time"],
                    "dimensions": {"width": int(original.shape[1]), "height": int(original.shape[0])},
                }
            )
        except ImageInputError as error:
            return json_error(str(error), 400)
        except ValueError as error:
            return json_error(str(error), 400)
        except ModelUnavailableError as error:
            return json_error(str(error), 503)
        except OSError as error:
            app.logger.exception("Could not save detection output")
            return json_error(f"Could not save processing output: {error}", 500)
        except Exception as error:
            app.logger.exception("YOLO object detection failed")
            return json_error(f"Object detection failed: {error}", 500)

    @app.post("/api/segment")
    def segment_route():
        try:
            original, original_path, results, annotated_path = run_detection_request()
            output_folder = Path(app.config["OUTPUT_FOLDER"])
            mask_items = []
            overlay = original.copy()
            for index, mask in enumerate(results["masks"]):
                color = np.array(
                    [(37 * (index + 2)) % 255, (97 * (index + 4)) % 255, (173 * (index + 1)) % 255],
                    dtype=np.uint8,
                )
                selected = mask > 0
                overlay[selected] = (
                    0.55 * overlay[selected].astype(np.float32) + 0.45 * color
                ).astype(np.uint8)
                mask_path = save_image(mask, output_folder, f"mask_{index + 1}")
                mask_items.append(
                    {
                        "url": file_url(mask_path),
                        "download": url_for("download_output", filename=mask_path.name),
                        "label": (
                            results["detections"][index]["label"]
                            if index < len(results["detections"])
                            else f"Instance {index + 1}"
                        ),
                        "confidence": (
                            results["detections"][index]["confidence"]
                            if index < len(results["detections"])
                            else None
                        ),
                    }
                )
            overlay_path = save_image(overlay, output_folder, "segmentation")
            return jsonify(
                {
                    "success": True,
                    "original": file_url(original_path),
                    "result": file_url(overlay_path),
                    "download": url_for("download_output", filename=overlay_path.name),
                    "annotated": file_url(annotated_path),
                    "objects": results["detections"],
                    "object_count": len(results["detections"]),
                    "masks": mask_items,
                    "processing_time": results["processing_time"],
                }
            )
        except ImageInputError as error:
            return json_error(str(error), 400)
        except ValueError as error:
            return json_error(str(error), 400)
        except ModelUnavailableError as error:
            return json_error(str(error), 503)
        except OSError as error:
            app.logger.exception("Could not save segmentation output")
            return json_error(f"Could not save processing output: {error}", 500)
        except Exception as error:
            app.logger.exception("YOLO instance segmentation failed")
            return json_error(f"Image segmentation failed: {error}", 500)

    @app.post("/api/analytics")
    def analytics_route():
        try:
            original, original_path = upload_image()
            gamma = float(request.form.get("gamma", "1.35"))
            enhanced = enhance_image(original, gamma)
            results = run_segmentation(
                original, Path(app.config["MODEL_PATH"]), confidence=0.25
            )
            histogram_path = save_histogram_comparison(
                original,
                enhanced,
                Path(app.config["OUTPUT_FOLDER"]),
                f"analytics_{uuid4().hex}.png",
            )
            enhanced_path = save_image(
                enhanced, Path(app.config["OUTPUT_FOLDER"]), "analytics_enhanced"
            )
            original_features, _ = extract_hog_features(original)
            enhanced_features, _ = extract_hog_features(enhanced)
            return jsonify(
                {
                    "success": True,
                    "original": file_url(original_path),
                    "enhanced": file_url(enhanced_path),
                    "histogram": file_url(histogram_path),
                    "object_count": len(results["detections"]),
                    "dimensions": {"width": int(original.shape[1]), "height": int(original.shape[0])},
                    "original_brightness": image_brightness(original),
                    "enhanced_brightness": image_brightness(enhanced),
                    "processing_time": results["processing_time"],
                    "hog_similarity": hog_cosine_similarity(
                        original_features, enhanced_features
                    ),
                }
            )
        except ImageInputError as error:
            return json_error(str(error), 400)
        except ValueError as error:
            return json_error(str(error), 400)
        except ModelUnavailableError as error:
            return json_error(str(error), 503)
        except OSError as error:
            app.logger.exception("Could not save analytics output")
            return json_error(f"Could not save analytics output: {error}", 500)
        except Exception as error:
            app.logger.exception("Image analytics failed")
            return json_error(f"Analytics processing failed: {error}", 500)

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
