"""
config.py — Konfigurasi Global ML Boostify
Semua nilai threshold, path, dan parameter ada di sini.
Kalau mau ubah sesuatu, cukup edit file ini saja.

Update:
- Model: dlib ResNet (via library face_recognition) — ringan, stabil di Raspi
- Detector: HOG (paling ringan, tanpa GPU)
- 50 foto + augmentasi 15x = 750 data per orang
"""

import os

# =============================================================
# PARAMETER SMILE DETECTION
# =============================================================
SMILE_VOTE_FRAMES   = 5    # jumlah frame untuk voting
SMILE_VOTE_THRESH   = 3    # minimal frame senyum → dianggap senyum
SMILE_MIN_NEIGHBORS = 8    # sensitivitas deteksi

# =============================================================
# PATH DATASET & MODEL
# =============================================================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_RAW     = os.path.join(BASE_DIR, "dataset", "raw")
DATASET_PROC    = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

# File output model
EMBEDDING_FILE  = os.path.join(MODEL_DIR, "embeddings.pkl")
LABEL_FILE      = os.path.join(MODEL_DIR, "labels.pkl")

# =============================================================
# PARAMETER KAMERA & FRAME
# =============================================================
CAMERA_INDEX    = 1          # 0 = kamera default, 1 = USB eksternal
FRAME_WIDTH     = 320        # resolusi capture
FRAME_HEIGHT    = 240
FRAME_SKIP      = 8          # proses 1 dari setiap N frame (hemat CPU Raspi)
COOLDOWN_SEC    = 60         # jeda detik setelah absen berhasil

# =============================================================
# PARAMETER PREPROCESSING
# =============================================================
CLAHE_CLIP      = 3.0        # semakin besar = kontras makin kuat
CLAHE_GRID      = (8, 8)     # ukuran tile CLAHE
FACE_SIZE       = (160, 160) # ukuran wajah setelah di-crop preprocess
MIN_FACE_SIZE   = 15         # piksel minimum agar wajah dianggap valid

# =============================================================
# PARAMETER RECOGNITION (dlib ResNet via face_recognition)
# =============================================================
# CATATAN: SEMANTIK BERUBAH!
# Sekarang confidence dihitung dari Euclidean distance dlib:
#   confidence = 1 - distance
# Jadi:
#   distance 0.0  → confidence 1.0 (sangat mirip)
#   distance 0.6  → confidence 0.4 (batas tolerance default dlib)
#   distance 1.0+ → confidence 0.0 (beda orang)
SIMILARITY_THRESHOLD  = 0.4   # confidence >= 0.4 → dikenali (distance <= 0.6)
                               # Naikkan (0.5) = lebih ketat
                               # Turunkan (0.3) = lebih longgar

MIN_PHOTOS_PER_PERSON = 50    # 50 foto per orang (balance kecepatan & akurasi)
AUGMENT_PER_IMAGE     = 15    # 50 foto × 15 = 750 data per orang

# =============================================================
# PARAMETER MODEL face_recognition
# =============================================================
# Model recognition: dlib ResNet (built-in face_recognition library, 128-dim)
MODEL_BACKEND    = "dlib_resnet"    # informasional saja, library cuma punya 1 model

# Detector face_recognition:
# "hog" → REKOMENDASI: ringan, CPU-only, cocok untuk Raspi
# "cnn" → lebih akurat tapi butuh GPU / lebih berat
DETECTOR_BACKEND = "hog"
FR_DETECTOR      = "hog"            # alias yg dipakai train.py & predict.py
FR_NUM_JITTERS   = 1                # 1 = cepat & cukup akurat; 10 = paling akurat tapi lambat

# =============================================================
# LOGGING
# =============================================================
LOG_FILE        = os.path.join(LOG_DIR, "boostify_ml.log")
LOG_LEVEL       = "INFO"