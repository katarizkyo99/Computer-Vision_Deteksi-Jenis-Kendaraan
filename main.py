'''Tahap Persiapan dan Setup'''

#Import Library yang diperlukan
from flask import Flask, request, render_template, send_from_directory
import os
import glob
import cv2
import numpy as np
from roboflow import Roboflow
from ultralytics import YOLO
import supervision as sv
from werkzeug.utils import secure_filename

# Load configuration from environment (secure defaults)
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY')
YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', 'yolov8x.pt')

# Optional: download dataset from Roboflow only if API key provided
if ROBOFLOW_API_KEY:
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("tanzim-mostafa").project("p2_dhaka_dataset-f6ba6")
        version = project.version(29)
        dataset = version.download("yolov8")
    except Exception as e:
        # If download fails, print a warning but continue — dataset is optional for inference
        print(f"Roboflow dataset download failed: {e}")

# Konfigurasi Flask dan Membuat folder untuk upload dan hasil deteksi
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


'''Training / Fine-Tuning Model'''

# Load model YOLO (use YOLO_WEIGHTS environment variable or fallback)
try:
    model = YOLO(YOLO_WEIGHTS)
except Exception as e:
    print(f"Failed to load YOLO model from '{YOLO_WEIGHTS}': {e}")
    raise


# Memberikan nama kelas untuk deteksi
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
# Map selected class names to IDs; ignore names not found
SELECTED_CLASS_IDS = []
for name in SELECTED_CLASS_NAMES:
    if name in CLASS_NAMES_DICT.values():
        # reverse lookup: name -> id
        SELECTED_CLASS_IDS.append({value: key for key, value in CLASS_NAMES_DICT.items()}[name])


'''Implementasi Deteksi & Klasifikasi'''

# Inisialisasi Annotator untuk memberikan anotasi pada gambar hasil deteksi
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

#Memproses gambar yang diunggah
def process_image(image_path):
    image = cv2.imread(image_path)
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(SELECTED_CLASS_IDS) > 0:
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    labels = [f"{model.model.names[class_id]} {confidence:.2f}" for confidence, class_id in zip(detections.confidence, detections.class_id)]
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, annotated_image)
    return os.path.basename(result_path), labels


'''Membangun Aplikasi Web'''

#Mendefinisikan laman untuk mengunggah file
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result_filename, labels = process_image(filepath)
            return render_template('index.html', filename=result_filename, labels=labels)
    return render_template('index.html', filename=None, labels=None)

#menampilkan gambar yang sudah diproses
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Note: this serves files from the results/ folder (annotated images)
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


#Menjalankan Flask
if __name__ == '__main__':
    # Keep debug for development; do NOT enable in production
    app.run(debug=True)
