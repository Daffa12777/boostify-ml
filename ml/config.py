"""
config.py — Konfigurasi Global ML Boostify
Semua nilai threshold, path, dan parameter ada di sini.
Kalau mau ubah sesuatu, cukup edit file ini saja.
"""

import os

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
FRAME_WIDTH     = 640        # resolusi capture
FRAME_HEIGHT    = 480
FRAME_SKIP      = 5          # proses 1 dari setiap N frame (hemat CPU Raspi)
COOLDOWN_SEC    = 3          # jeda detik setelah absen berhasil

# =============================================================
# PARAMETER PREPROCESSING
# =============================================================
CLAHE_CLIP      = 3.0        # semakin besar = kontras makin kuat
CLAHE_GRID      = (8, 8)     # ukuran tile CLAHE
FACE_SIZE       = (160, 160) # ukuran wajah setelah di-crop (input model)
MIN_FACE_SIZE   = 20         # piksel minimum agar wajah dianggap valid

# =============================================================
# PARAMETER RECOGNITION
# =============================================================
SIMILARITY_THRESHOLD = 0.55  # cosine similarity >= ini → dikenali
                              # Naikkan (0.70) = lebih ketat
                              # Turunkan (0.50) = lebih longgar
MIN_PHOTOS_PER_PERSON = 10   # minimal foto per orang untuk training
AUGMENT_PER_IMAGE = 10        # jumlah augmentasi per foto asli

# =====================================
# ========================
# PARAMETER TRAINING
# =============================================================
MODEL_BACKEND   = "Facenet"  # pilihan: "ArcFace", "Facenet", "Facenet512"
DETECTOR_BACKEND = "mtcnn"   # pilihan: "mtcnn", "retinaface", "opencv"

# =============================================================
# LOGGING
# =============================================================
LOG_FILE        = os.path.join(LOG_DIR, "boostify_ml.log")
LOG_LEVEL       = "INFO"
