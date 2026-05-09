"""
train.py — Training & Generate Embeddings Boostify
====================================================
Fungsi:
  1. Baca foto dari dataset/processed/[nama]/
  2. Extract embedding (512-dim vector) tiap wajah pakai DeepFace + ArcFace
  3. Rata-ratakan embedding per orang
  4. Simpan ke models/embeddings.pkl dan models/labels.pkl

Cara pakai:
  python train.py

Output:
  models/embeddings.pkl  → dict {nama: embedding_vector}
  models/labels.pkl      → list nama orang yang terdaftar
  
Catatan:
  - Jalankan preprocess.py DULU sebelum train.py
  - Training cukup dilakukan di laptop/PC (bukan di Raspi)
  - Setelah training, copy folder models/ ke Raspi
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_PROC, MODEL_DIR,
    EMBEDDING_FILE, LABEL_FILE,
    MODEL_BACKEND, DETECTOR_BACKEND,
    MIN_PHOTOS_PER_PERSON
)
from utils.logger import get_logger

logger = get_logger("train")


# ─────────────────────────────────────────────
# LAZY IMPORT DeepFace (berat, load sekali saja)
# ─────────────────────────────────────────────
def load_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        logger.error("DeepFace belum terinstall! Jalankan: pip install deepface")
        sys.exit(1)


# ─────────────────────────────────────────────
# EXTRACT EMBEDDING DARI SATU GAMBAR
# ─────────────────────────────────────────────
def extract_embedding(DeepFace, image_path: str) -> np.ndarray | None:
    """
    Ekstrak embedding vector dari satu file gambar.
    
    Pakai model ArcFace → menghasilkan vektor 512 dimensi.
    Vektor ini adalah 'sidik jari digital' dari wajah tersebut.
    
    Return: numpy array (512,) atau None jika gagal
    """
    try:
        result = DeepFace.represent(
            img_path        = image_path,
            model_name      = MODEL_BACKEND,
            detector_backend= DETECTOR_BACKEND,
            enforce_detection = False,  # tidak error jika wajah kurang jelas
            align           = True      # align wajah sebelum extract
        )
        if result and len(result) > 0:
            embedding = np.array(result[0]["embedding"])
            return embedding
        return None

    except Exception as e:
        logger.debug(f"Gagal extract {image_path}: {e}")
        return None


# ─────────────────────────────────────────────
# TRAINING PIPELINE UTAMA
# ─────────────────────────────────────────────
def train():
    """
    Loop semua orang di dataset/processed/,
    kumpulkan semua embedding-nya, rata-ratakan,
    simpan ke file .pkl.
    """
    if not os.path.exists(DATASET_PROC):
        logger.error(f"Folder processed tidak ditemukan: {DATASET_PROC}")
        logger.info("Jalankan dulu: python preprocess.py")
        return

    persons = [d for d in os.listdir(DATASET_PROC)
               if os.path.isdir(os.path.join(DATASET_PROC, d))]

    if len(persons) == 0:
        logger.error("Tidak ada data di dataset/processed/")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Loading model {MODEL_BACKEND} (pertama kali mungkin download ~100MB) ...")

    DeepFace = load_deepface()

    embeddings_db = {}   # { "nama": np.array (512,) }
    labels        = []

    logger.info(f"\nMulai training untuk {len(persons)} orang ...\n")

    for person in persons:
        proc_dir = os.path.join(DATASET_PROC, person)
        photo_files = [
            os.path.join(proc_dir, f)
            for f in os.listdir(proc_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(photo_files) < MIN_PHOTOS_PER_PERSON:
            logger.warning(f"[{person}] Foto kurang ({len(photo_files)}), dilewati.")
            continue

        logger.info(f"[{person}] Mengekstrak embedding dari {len(photo_files)} foto ...")

        all_embeddings = []
        failed = 0

        for fpath in tqdm(photo_files, desc=f"  {person}"):
            emb = extract_embedding(DeepFace, fpath)
            if emb is not None:
                all_embeddings.append(emb)
            else:
                failed += 1

        if len(all_embeddings) == 0:
            logger.warning(f"[{person}] Semua foto gagal di-extract, dilewati.")
            continue

        # Rata-ratakan semua embedding → 1 vektor representasi per orang
        mean_embedding = np.mean(all_embeddings, axis=0)

        # L2 normalisasi → penting untuk cosine similarity
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        embeddings_db[person] = mean_embedding
        labels.append(person)

        logger.info(
            f"  [{person}] ✅ Berhasil: {len(all_embeddings)} | "
            f"❌ Gagal: {failed} | "
            f"Embedding shape: {mean_embedding.shape}"
        )

    if len(embeddings_db) == 0:
        logger.error("Tidak ada embedding yang berhasil dibuat.")
        return

    # ─── Simpan ke file ───
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(embeddings_db, f)

    with open(LABEL_FILE, "wb") as f:
        pickle.dump(labels, f)

    logger.info(f"\n{'='*50}")
    logger.info(f"✅ TRAINING SELESAI")
    logger.info(f"Total orang terdaftar : {len(labels)}")
    logger.info(f"Nama terdaftar        : {labels}")
    logger.info(f"File embedding        : {EMBEDDING_FILE}")
    logger.info(f"File labels           : {LABEL_FILE}")
    logger.info(f"{'='*50}")
    logger.info("Sekarang copy folder models/ ke Raspberry Pi.")


# ─────────────────────────────────────────────
# TAMBAH SATU ORANG BARU (tanpa re-train semua)
# ─────────────────────────────────────────────
def register_new_person(nama: str):
    """
    Daftarkan satu orang baru ke database embedding
    tanpa perlu re-training semua orang dari awal.
    
    Cara pakai:
      python train.py --register "Nama Orang"
    """
    proc_dir = os.path.join(DATASET_PROC, nama)
    if not os.path.exists(proc_dir):
        logger.error(f"Folder tidak ditemukan: {proc_dir}")
        logger.info(f"Buat folder & isi foto dulu, lalu jalankan preprocess.py untuk {nama}")
        return

    # Load database yang sudah ada
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

    logger.info(f"Mendaftarkan [{nama}] dari {len(photo_files)} foto ...")
    DeepFace = load_deepface()

    all_embeddings = []
    for fpath in tqdm(photo_files, desc=f"  {nama}"):
        emb = extract_embedding(DeepFace, fpath)
        if emb is not None:
            all_embeddings.append(emb)

    if len(all_embeddings) == 0:
        logger.error(f"Tidak ada embedding berhasil untuk {nama}.")
        return

    mean_embedding = np.mean(all_embeddings, axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm

    embeddings_db[nama] = mean_embedding
    if nama not in labels:
        labels.append(nama)

    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(embeddings_db, f)
    with open(LABEL_FILE, "wb") as f:
        pickle.dump(labels, f)

    logger.info(f"✅ [{nama}] berhasil didaftarkan!")
    logger.info(f"Total terdaftar: {len(labels)} orang")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boostify ML Training")
    parser.add_argument("--register", type=str, default=None,
                        help="Nama orang yang ingin didaftarkan ulang/baru")
    args = parser.parse_args()

    if args.register:
        register_new_person(args.register)
    else:
        train()
