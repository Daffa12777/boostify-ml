"""
train.py — Training & Generate Embeddings Boostify
Model: dlib ResNet via library face_recognition (128-dim)
Output: models/embeddings.pkl + models/labels.pkl

Format output SAMA dengan versi GhostFaceNet:
  embeddings.pkl → dict {nama: mean_encoding_128d}
  labels.pkl     → list nama
Jadi predict.py & predict_thread.py tidak perlu diubah strukturnya.
"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import face_recognition

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_PROC, MODEL_DIR,
    EMBEDDING_FILE, LABEL_FILE,
    MIN_PHOTOS_PER_PERSON,
    FR_DETECTOR, FR_NUM_JITTERS
)
from utils.logger import get_logger

logger = get_logger("train")


# ─────────────────────────────────────────────
# EXTRACT EMBEDDING (dlib ResNet via face_recognition)
# ─────────────────────────────────────────────
def extract_embedding(image_path: str):
    """
    Load gambar, deteksi wajah pakai HOG, ekstrak encoding 128-d.
    Fallback: kalau HOG gagal deteksi (umum di crop ketat hasil preprocess),
    pakai seluruh gambar sebagai bbox wajah → encoding tetap bisa diambil.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # face_recognition butuh RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Coba deteksi pakai HOG dulu
    locations = face_recognition.face_locations(rgb, model=FR_DETECTOR)

    # Fallback: kalau nggak ada wajah terdeteksi, anggap seluruh gambar = wajah
    # (works karena preprocess sudah crop ketat ke area wajah)
    if not locations:
        h, w = rgb.shape[:2]
        locations = [(0, w, h, 0)]   # format face_recognition: (top, right, bottom, left)

    # Pilih wajah terbesar
    locations = sorted(
        locations,
        key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]),
        reverse=True
    )

    encodings = face_recognition.face_encodings(
        rgb,
        known_face_locations=locations[:1],
        num_jitters=FR_NUM_JITTERS
    )

    if not encodings:
        return None

    return encodings[0]   # 128-d numpy array


# ─────────────────────────────────────────────
# TRAIN — full retrain
# ─────────────────────────────────────────────
def train():
    if not os.path.exists(DATASET_PROC):
        logger.error(f"Folder processed tidak ditemukan: {DATASET_PROC}")
        logger.info("Jalankan dulu: python preprocess.py")
        return

    persons = [d for d in os.listdir(DATASET_PROC)
               if os.path.isdir(os.path.join(DATASET_PROC, d))]

    if not persons:
        logger.error("Tidak ada data di dataset/processed/")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Loading model dlib ResNet (face_recognition) ...")
    logger.info(f"Detector: {FR_DETECTOR} | num_jitters: {FR_NUM_JITTERS}")

    embeddings_db = {}
    labels        = []

    logger.info(f"Training untuk {len(persons)} orang ...\n")

    for person in persons:
        proc_dir    = os.path.join(DATASET_PROC, person)
        photo_files = [
            os.path.join(proc_dir, f)
            for f in os.listdir(proc_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(photo_files) < MIN_PHOTOS_PER_PERSON:
            logger.warning(f"[{person}] Foto kurang ({len(photo_files)}), dilewati.")
            continue

        logger.info(f"[{person}] Ekstrak encoding dari {len(photo_files)} foto ...")

        all_embeddings = []

        for fpath in tqdm(photo_files, desc=f"  {person}"):
            emb = extract_embedding(fpath)
            if emb is not None:
                all_embeddings.append(emb)

        if not all_embeddings:
            logger.warning(f"[{person}] Semua foto gagal, dilewati.")
            continue

        # Rata-rata encoding → jadi 1 representasi per orang
        # (Tidak di-L2-normalize karena dlib pakai Euclidean, bukan cosine)
        mean_embedding = np.mean(all_embeddings, axis=0)

        embeddings_db[person] = mean_embedding
        labels.append(person)

        logger.info(f"  [{person}] ✅ {len(all_embeddings)} encoding | shape: {mean_embedding.shape}")

    if not embeddings_db:
        logger.error("Tidak ada embedding yang berhasil dibuat.")
        return

    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(embeddings_db, f)

    with open(LABEL_FILE, "wb") as f:
        pickle.dump(labels, f)

    logger.info(f"\n{'='*50}")
    logger.info(f"✅ TRAINING SELESAI")
    logger.info(f"Total orang  : {len(labels)}")
    logger.info(f"Terdaftar    : {labels}")
    logger.info(f"Embedding    : {EMBEDDING_FILE}")
    logger.info(f"Labels       : {LABEL_FILE}")
    logger.info(f"{'='*50}")


# ─────────────────────────────────────────────
# REGISTER — tambah 1 orang baru tanpa retrain semua
# ─────────────────────────────────────────────
def register_new_person(nama: str):
    proc_dir = os.path.join(DATASET_PROC, nama)
    if not os.path.exists(proc_dir):
        logger.error(f"Folder tidak ditemukan: {proc_dir}")
        return

    # Load DB yang ada (kalau belum ada, mulai kosong)
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            embeddings_db = pickle.load(f)
        with open(LABEL_FILE, "rb") as f:
            labels = pickle.load(f)
    else:
        embeddings_db = {}
        labels        = []

    photo_files = [
        os.path.join(proc_dir, f)
        for f in os.listdir(proc_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    logger.info(f"Register [{nama}] dari {len(photo_files)} foto ...")

    all_embeddings = []
    for fpath in tqdm(photo_files, desc=f"  {nama}"):
        emb = extract_embedding(fpath)
        if emb is not None:
            all_embeddings.append(emb)

    if not all_embeddings:
        logger.error(f"Tidak ada embedding untuk {nama}.")
        return

    mean_embedding = np.mean(all_embeddings, axis=0)

    embeddings_db[nama] = mean_embedding
    if nama not in labels:
        labels.append(nama)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(embeddings_db, f)
    with open(LABEL_FILE, "wb") as f:
        pickle.dump(labels, f)

    logger.info(f"✅ [{nama}] berhasil didaftarkan!")
    logger.info(f"Total terdaftar: {len(labels)} orang")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", type=str, default=None,
                        help="Nama orang baru (tambah ke DB tanpa retrain semua)")
    args = parser.parse_args()

    if args.register:
        register_new_person(args.register)
    else:
        train()