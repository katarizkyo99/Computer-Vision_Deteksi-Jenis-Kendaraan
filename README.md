# Vehicle Type Detection (Computer Vision)

A simple Flask web application that performs vehicle type detection (car, motorcycle, bus, truck) on uploaded images using YOLO (Ultralytics) and annotated with the supervision library. The project is intended as a local demo / research prototype for developers, students, and researchers working on object detection.

## Features
- Upload an image through a web UI and get a detection result image with bounding boxes and labels.
- Uses a YOLOv8 model (Ultralytics) for inference.
- Optional: download dataset from Roboflow if ROBOFLOW_API_KEY is provided.

## Stack
- Language: Python
- Web framework: Flask
- Notable libraries: ultralytics (YOLO), roboflow (dataset download), OpenCV (cv2), supervision

## Requirements
- Python 3.8+
- GPU recommended for real-time/inference performance but CPU works for small demos.

## Installation
1. Clone the repository

```bash
git clone https://github.com/katarizkyo99/Computer-Vision_Deteksi-Jenis-Kendaraan.git
cd Computer-Vision_Deteksi-Jenis-Kendaraan
```

2. (Recommended) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows (PowerShell)
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Prepare model weights

- By default the application tries to load `yolov8x.pt` from the working directory. Set `YOLO_WEIGHTS` environment variable to a path to your model file if you use a different file.

5. (Optional) Provide Roboflow API key

- If you want the project to automatically download the dataset (as the original code attempted), set `ROBOFLOW_API_KEY` in your environment. See `.env.example` for variable names.

## Configuration
Environment variables supported (see `.env.example`):

- `ROBOFLOW_API_KEY` — (optional) API key used to download a dataset from Roboflow.
- `YOLO_WEIGHTS` — path to YOLO model weights. Defaults to `yolov8x.pt`.

Example using a local `.env` (not committed to the repository):

```bash
export ROBOFLOW_API_KEY="<your_key_here>"
export YOLO_WEIGHTS="yolov8x.pt"
```

On Windows (PowerShell):

```powershell
$env:ROBOFLOW_API_KEY = "<your_key_here>"
$env:YOLO_WEIGHTS = "yolov8x.pt"
```

## Running the application
Start the Flask development server:

```bash
python main.py
```

The app will start on http://127.0.0.1:5000/. Upload an image via the form and the annotated image will be shown after inference.

## Project structure
```
README.md
main.py                  # Flask application + inference pipeline
templates/index.html     # Upload form + result display
foto/                    # sample images
uploads/                 # runtime: uploaded images (gitignored)
results/                 # runtime: annotated result images (gitignored)
runs/                    # run outputs (gitignored)
```
