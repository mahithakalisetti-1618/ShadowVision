# ShadowVision

ShadowVision is a local Flask computer vision application for improving poorly lit images and inspecting image-processing outputs alongside YOLO object detection and instance segmentation. It uses real OpenCV, NumPy, scikit-image, scikit-learn, Matplotlib, and Ultralytics processing; it does not use a paid image API or fabricate model results.

## Problem statement

Dark shadows and uneven illumination can reduce contrast, hide boundaries, and make objects harder to inspect. This project presents interpretable classical image processing and pretrained instance segmentation in one workflow so their outputs can be compared on the same input.

## Objectives

- Improve and compare image brightness and luminance contrast.
- Explore spatial smoothing, sharpening, edge detection, and adaptive thresholding.
- Extract and visualize Histogram of Oriented Gradients (HOG) features.
- Detect objects and generate per-instance masks using a pretrained YOLO segmentation model.
- Report image measurements and visualize histogram changes.

## Features

- Drag/drop and file-picker uploads for JPG, JPEG, PNG, and BMP (up to 12 MB).
- Gamma correction, luminance histogram equalization, Gaussian/median/bilateral smoothing, and sharpening.
- Canny and Sobel edge maps, adaptive Gaussian thresholding, and HOG visualization.
- YOLO bounding boxes, labels, confidence values, instance masks, and downloadable outputs.
- Matplotlib color-histogram comparisons and brightness/dimension/inference measurements.
- Responsive pages for Overview, Enhance, Detect, Segment, Analytics, and About.
- Local server processing. Detection and segmentation require local model weights; missing weights return a clear error.

## Technologies and computer vision methods

| Method | Implementation |
|---|---|
| Histogram processing | Per-channel intensity histograms, drawn and compared with Matplotlib |
| Histogram equalization | OpenCV equalization of the Y (luminance) channel in YCrCb |
| Intensity transformation | Vectorized gamma lookup table |
| Spatial filtering | Gaussian, median, bilateral, and a 3×3 sharpening kernel |
| Edge detection | Canny and Sobel gradient magnitude |
| Adaptive thresholding | Gaussian adaptive threshold on grayscale |
| HOG | scikit-image HOG descriptor and visualization; scikit-learn cosine similarity for descriptors |
| Object detection | Ultralytics YOLO segmentation model, class/confidence/box output |
| Instance segmentation | YOLO per-instance masks, binary-mask downloads, and colored overlay |

## Project structure

```text
ShadowVision/
├── app.py
├── config.py
├── enhancement.py
├── detection.py
├── feature_extraction.py
├── analytics.py
├── image_utils.py
├── requirements.txt
├── README.md
├── .env.example
├── models/
│   └── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── analysis.html
│   ├── detection.html
│   ├── segmentation.html
│   ├── analytics.html
│   ├── about.html
│   ├── 404.html
│   └── error.html
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── uploads/
│   └── outputs/
└── tests/
    ├── test_enhancement.py
    ├── test_image_utils.py
    └── test_routes.py
```

Uploaded images are assigned generated filenames in `static/uploads/`. Processed images and chart files go to `static/outputs/`; both directories are ignored by Git.

## Setup on Windows

Open PowerShell in the project directory (the directory containing `app.py`). Python 3.10 or newer is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, either enable local scripts for your user with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### YOLO segmentation model setup

The default model path is `models\yolo11n-seg.pt`. The application deliberately does not download weights when a request arrives. To fetch the official Ultralytics model using the installed package, run:

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
Push-Location .\models
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')"
Pop-Location
Test-Path .\models\yolo11n-seg.pt
```

The first command that initializes `YOLO('yolo11n-seg.pt')` downloads weights if they are not already present and network access is available. Model downloads are large and may take a few minutes. The final `Test-Path` should print `True`. You can instead set `SHADOWVISION_MODEL` to an existing compatible local `.pt` segmentation model before launching:

```powershell
$env:SHADOWVISION_MODEL = "D:\models\my-segmentation-model.pt"
```

Detection and segmentation return an explicit HTTP 503 error with the configured model path if the file is absent or unreadable.

## Run the application

From the project directory, with the virtual environment active:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser. For local development, Flask debug mode is enabled. Do not expose the development server publicly.

## Using the pages

1. **Overview** — review the purpose, capabilities, and links to the processing tools.
2. **Enhance** — upload an image; adjust gamma, optional luminance equalization, a spatial filter, and edge method. Apply the operations to inspect the enhanced image, edge output, adaptive threshold, HOG visualization, histogram, and computed metrics. Download the processed PNG or reset the form.
3. **Detect** — upload an image and set confidence. The configured YOLO model returns actual object labels, scores, boxes, annotated output, count, and inference time.
4. **Segment** — inspect the original, combined color mask overlay, each binary instance mask, and detected classes; download the overlay or individual masks.
5. **Analytics** — choose an image and gamma value. The page computes grayscale-mean brightness, image dimensions, YOLO detection count and inference time, HOG cosine similarity, and a Matplotlib histogram comparison.
6. **About** — read the project problem statement, objectives, algorithm methodology, technologies, applications, and future scope.

If YOLO weights are missing, image enhancement remains available; detection, segmentation, and analytics report the model setup error rather than showing invented results.

## Tests

Install the requirements, then run the test suite:

```powershell
python -m pytest
```

Tests cover image transformations, upload validation, page routing, enhancement outputs, and the missing-model response. They do not download model weights or require a GPU.

## Troubleshooting

- **`ModuleNotFoundError`** — activate the virtual environment and run `python -m pip install -r requirements.txt` using that same interpreter.
- **Model not found** — confirm `models\yolo11n-seg.pt` exists, or set `SHADOWVISION_MODEL` to the absolute path of your segmentation weights.
- **Model download fails** — verify network access, rerun the documented download command from `models`, and check `Test-Path`.
- **Upload rejected** — use a readable JPG, JPEG, PNG, or BMP file under 12 MB and 40 million pixels.
- **Slow inference** — the default small model runs on the available device; CPU inference may take longer. The reported duration is measured for model prediction.
- **Port 5000 is occupied** — stop the other local service or change the `app.run` port in `app.py`.

## Local API

- `POST /api/enhance` — multipart form fields: `image`, `gamma`, `equalize`, `filter`, `edge_method`.
- `POST /api/detect` — multipart form fields: `image`, optional `confidence`.
- `POST /api/segment` — multipart form fields: `image`, optional `confidence`.
- `POST /api/analytics` — multipart form fields: `image`, optional `gamma`.

Processing errors use JSON responses with an explanatory `error` field. Enhancement validation errors use HTTP 400; absent model weights use HTTP 503; oversized uploads use HTTP 413.
