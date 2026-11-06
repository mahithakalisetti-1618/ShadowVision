# Model weights

The application expects a local Ultralytics YOLO segmentation weights file at `models/yolo11n-seg.pt` by default. From the project root, fetch the official pretrained weights with:

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
Push-Location .\models
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')"
Pop-Location
```

The first initialization downloads the model when network access is available. Model weights are excluded from Git. To use different compatible weights, set `SHADOWVISION_MODEL` to their full path. The app does not silently download or substitute a model during inference.
