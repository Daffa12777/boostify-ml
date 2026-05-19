"""
config.py — Konfigurasi Global ML Boostify
Semua nilai threshold, path, dan parameter ada di sini.
Kalau mau ubah sesuatu, cukup edit file ini saja.

Update:
- Model: GhostFaceNet (ringan + akurat untuk sedikit foto)
- Detector: opencv (lebih ringan dari mtcnn)
- 50 foto + augmentasi 15x = 750 data per orang
"""

import os

# =============================================================
# PARAMETER SMILE DETECTION — TAMBAHKAN INI
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
FACE_SIZE       = (160, 160) # ukuran wajah setelah di-crop (input model)
MIN_FACE_SIZE   = 15         # piksel minimum agar wajah dianggap valid

# =============================================================
# PARAMETER RECOGNITION
# =============================================================
SIMILARITY_THRESHOLD  = 0.55  # cosine similarity >= ini → dikenali
                               # Naikkan (0.70) = lebih ketat
                               # Turunkan (0.50) = lebih longgar
MIN_PHOTOS_PER_PERSON = 50    # 50 foto per orang (balance kecepatan & akurasi)
AUGMENT_PER_IMAGE     = 15    # 50 foto × 15 = 750 data per orang

# =============================================================
# PARAMETER TRAINING
# =============================================================
# Pilihan model:
# "GhostFaceNet" → REKOMENDASI: ringan + akurat untuk sedikit foto
# "ArcFace"      → akurasi tinggi tapi berat di Raspi
# "Facenet"      → ringan tapi kurang akurat untuk sedikit foto
# "Facenet512"   → Facenet versi lebih akurat, embedding 512 dim

MODEL_BACKEND    = "GhostFaceNet"  # ← ringan + akurat untuk Raspberry Pi

# Pilihan detector:
# "opencv"      → REKOMENDASI: paling ringan, cocok untuk Raspi
# "mtcnn"       → akurat tapi berat
# "retinaface"  → paling akurat tapi paling berat
DETECTOR_BACKEND = "opencv"        # ← ringan untuk Raspberry Pi

# =============================================================
# LOGGING
# =============================================================
LOG_FILE        = os.path.join(LOG_DIR, "boostify_ml.log")
LOG_LEVEL       = "INFO"