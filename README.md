# Vehicle Type Detection (YOLOv8 + Flask)

A simple Flask web application to detect vehicle types (car, motorcycle, bus, truck) in uploaded images using YOLOv8 (Ultralytics) and annotate results with supervision. The app provides a single-page web UI to upload images and returns annotated images with bounding boxes and class labels.

## Features

- Upload an image through a web UI and receive an annotated image with bounding boxes and labels.
- Uses Ultralytics YOLOv8 for inference.
- Annotates results using the supervision library.
- Includes a small collection of sample images (foto/).

## System Architecture

This repository hosts a single-process Flask application:

- main.py: starts a Flask server, loads the YOLO model, provides an upload form, runs inference and saves annotated results to results/.
- templates/index.html: simple HTML upload form and result view.
- foto/: sample images you can test with.

Runtime flow:
1. HTTP GET / -> serve upload form.
2. HTTP POST / (multipart form) -> save file to uploads/, run process_image(image_path) which:
   - runs YOLO inference,
   - filters detections for the selected classes (car, motorcycle, bus, truck),
   - annotates image (bounding box + labels),
   - writes annotated image to results/.
3. HTTP GET /uploads/<filename> -> serves annotated image from results/.

## Tech Stack

- Backend: Python, Flask
- ML / Inference: ultralytics (YOLOv8)
- Image processing: OpenCV (cv2), numpy
- Annotation: supervision
- Data download (optional): roboflow
- Templates: simple HTML (templates/index.html)

## Project Structure

```
project/
├── main.py                  # Flask app, model load, inference pipeline
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html           # upload form + result display
├── foto/                    # sample images
├── uploads/                 # runtime: uploaded images (gitignored)
├── results/                 # runtime: annotated result images (gitignored)
├── runs/                    # ultralytics run outputs (gitignored)
└── README.md
```

## Requirements

- Python 3.8+
- pip
- A YOLOv8 model weights file (the code calls `YOLO("yolov8x.pt")` — place your weights at this path or modify main.py).
- Optional: GPU and CUDA drivers for faster inference (CPU will work for small demos).
- See requirements.txt for Python packages:
  - Flask, numpy, opencv-python-headless, roboflow, ultralytics, supervision, werkzeug

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/katarizkyo99/Computer-Vision_Deteksi-Jenis-Kendaraan.git
   cd Computer-Vision_Deteksi-Jenis-Kendaraan
   ```
2. (Recommended) Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate    # Windows (PowerShell)
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Put YOLO weights file in repository root (default expected filename: `yolov8x.pt`) or edit main.py to point to your weights.

## Configuration

Notes about configuration observed in code:

- main.py currently hardcodes:
  - Roboflow API key (in code) and a specific Roboflow project download call.
  - Model path `"yolov8x.pt"` is hardcoded in `model = YOLO("yolov8x.pt")`.

Recommended (safer) approach:

- Do NOT store API keys in source. Remove hardcoded keys and use environment variables (e.g., ROBOFLOW_API_KEY).
- You can set environment variables (shell example):
  ```bash
  export ROBOFLOW_API_KEY="<your_key_here>"
  export YOLO_WEIGHTS="yolov8x.pt"
  ```
  Then modify main.py to read these (not done in current code).

## Running the Project

Start the Flask development server:

```bash
python main.py
```

Open http://127.0.0.1:5000/ in your browser, upload an image, and wait for the annotated result.

### Endpoints (as implemented)

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | / | Upload form (templates/index.html) |
| POST | / | Upload image file (multipart/form-data, field name `file`); returns index.html with result filename and labels |
| GET | /uploads/<filename> | Returns the annotated image from results/ (note: route is named '/uploads' but serves from results/) |

## Usage

- Use the web form to upload a JPEG/PNG image.
- After upload, the server runs YOLO inference, writes an annotated copy to results/, and displays it on the page along with detected labels.

## Database

- No database is used.

## Docker

- No Dockerfile or Docker Compose present in the repository.

## Troubleshooting

- FileNotFoundError for model: ensure `yolov8x.pt` exists in repository root or update main.py to the correct path.
- Roboflow errors: main.py contains a Roboflow download call which requires a valid API key; if you don't intend to use Roboflow, comment out that block.
- CV2 / OpenCV errors on some systems: if GUI-related errors occur, use the headless package already listed (opencv-python-headless).
- Dependency version conflicts: use a fresh virtualenv and pip install -r requirements.txt.

## Security Notes

- main.py previously contained a hardcoded Roboflow API key and calls Roboflow downloading code at startup. This is a credential exposure risk. Remove hardcoded keys and load sensitive values from environment variables or a .env file (never commit secrets).
- The application runs the Flask development server with debug=True; do not use debug mode in production.

## Development

- To change detected classes or class filtering, check SELECTED_CLASS_NAMES in main.py.
- To change model path, update the YOLO(...) argument in main.py or refactor main.py to read YOLO_WEIGHTS from environment.

## Contributing

- This is a small personal project. If you want to contribute, open an issue or submit a PR; follow the code style already present (simple Python scripts + HTML).

## License

- No LICENSE file found in repository. Do not assume a license; add one if you intend to make the project public under specific terms.

## Author

- Repository owner: katarizkyo99
