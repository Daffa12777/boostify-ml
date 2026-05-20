"""
train.py — Boostify Face Recognition Trainer (LBPH Version)
===========================================================

VERSI:
✅ Ringan untuk Raspberry Pi 5
✅ Tanpa DeepFace
✅ Tanpa TensorFlow
✅ Compatible Python 3.13
✅ Bisa recognize wajah
✅ Bisa tampil kode asisten
✅ Bisa connect Supabase
✅ Cepat realtime

OUTPUT:
models/trainer.yml
models/labels.pkl

STRUKTUR DATASET:
dataset/raw/
    Daffa/
        1.jpg
        2.jpg
    Nabila/
        1.jpg
        2.jpg

CARA TRAIN:
python3 train.py
"""

import os
import cv2
import pickle
import numpy as np
from datetime import datetime

# =========================================================
# CONFIG PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "raw")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

TRAINER_FILE = os.path.join(MODEL_DIR, "trainer.yml")
LABEL_FILE   = os.path.join(MODEL_DIR, "labels.pkl")

# =========================================================
# BUAT FOLDER JIKA BELUM ADA
# =========================================================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================================================
# LOGGER SEDERHANA
# =========================================================
def log(message):
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{waktu}] {message}")

# =========================================================
# LOAD FACE DETECTOR
# =========================================================
log("Loading Haarcascade Face Detector...")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    log("ERROR: Haarcascade gagal dimuat!")
    exit()

# =========================================================
# INIT LBPH RECOGNIZER
# =========================================================
log("Membuat LBPH Face Recognizer...")

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

# =========================================================
# STORAGE TRAINING
# =========================================================
faces = []
labels = []
label_map = {}

current_id = 0

# =========================================================
# VALIDASI DATASET
# =========================================================
if not os.path.exists(DATASET_DIR):
    log(f"ERROR: Folder dataset tidak ditemukan!")
    log(DATASET_DIR)
    exit()

persons = [
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
]

if len(persons) == 0:
    log("ERROR: Tidak ada folder orang di dataset/raw/")
    exit()

# =========================================================
# MULAI TRAINING
# =========================================================
log("=" * 50)
log("BOOSTIFY LBPH TRAINING")
log("=" * 50)

for person_name in persons:

    person_dir = os.path.join(DATASET_DIR, person_name)

    log(f"\n📂 Processing: {person_name}")

    # Mapping ID → Nama
    label_map[current_id] = person_name

    image_files = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_files) == 0:
        log(f"WARNING: Tidak ada foto untuk {person_name}")
        current_id += 1
        continue

    success_count = 0
    failed_count  = 0

    for image_name in image_files:

        image_path = os.path.join(person_dir, image_name)

        img = cv2.imread(image_path)

        if img is None:
            failed_count += 1
            continue

        # Convert ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect wajah
        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(detected_faces) == 0:
            failed_count += 1
            continue

        # Ambil wajah terbesar
        largest_face = max(
            detected_faces,
            key=lambda rect: rect[2] * rect[3]
        )

        (x, y, w, h) = largest_face

        face_crop = gray[y:y+h, x:x+w]

        # Resize standar
        face_crop = cv2.resize(face_crop, (200, 200))

        # Histogram Equalization
        face_crop = cv2.equalizeHist(face_crop)

        # Simpan face dan label
        faces.append(face_crop)
        labels.append(current_id)

        success_count += 1

    log(
        f"✅ Success: {success_count} | "
        f"❌ Failed: {failed_count}"
    )

    current_id += 1

# =========================================================
# VALIDASI HASIL
# =========================================================
if len(faces) == 0:
    log("ERROR: Tidak ada wajah berhasil diproses!")
    exit()

# =========================================================
# TRAIN MODEL
# =========================================================
log("\n🧠 Training model LBPH...")

recognizer.train(
    faces,
    np.array(labels)
)

# =========================================================
# SAVE MODEL
# =========================================================
log("💾 Menyimpan model...")

recognizer.save(TRAINER_FILE)

with open(LABEL_FILE, "wb") as f:
    pickle.dump(label_map, f)

# =========================================================
# SUMMARY
# =========================================================
log("\n" + "=" * 50)
log("✅ TRAINING SELESAI")
log("=" * 50)

log(f"👥 Total orang   : {len(label_map)}")
log(f"📸 Total wajah   : {len(faces)}")
log(f"💾 Trainer saved : {TRAINER_FILE}")
log(f"💾 Labels saved  : {LABEL_FILE}")

log("\n📋 LABEL MAP:")

for idx, name in label_map.items():
    log(f"   {idx} → {name}")

log("=" * 50)
log("BOOSTIFY TRAINING BERHASIL 🔥")
log("=" * 50)