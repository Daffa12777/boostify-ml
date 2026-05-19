"""
preprocess.py — Preprocessing Dataset Boostify
Fix: Hapus dead code setelah return None
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_RAW, DATASET_PROC,
    CLAHE_CLIP, CLAHE_GRID,
    FACE_SIZE, MIN_FACE_SIZE,
    AUGMENT_PER_IMAGE, MIN_PHOTOS_PER_PERSON
)
from utils.logger import get_logger

logger = get_logger("preprocess")


# ─────────────────────────────────────────────
# 1. CLAHE ENHANCEMENT
# ─────────────────────────────────────────────
def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# 2. DETEKSI & CROP WAJAH (FIXED — no dead code)
# ─────────────────────────────────────────────
def detect_and_crop_face(image: np.ndarray) -> np.ndarray | None:
    """
    Deteksi wajah dari gambar, crop & resize ke FACE_SIZE.
    Coba beberapa parameter dari yang paling longgar.
    Return None kalau tidak ada wajah terdeteksi.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Coba berbagai parameter dari longgar ke ketat
    for scale in [1.05, 1.1, 1.2]:
        for neighbors in [2, 3, 5]:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor  = scale,
                minNeighbors = neighbors,
                minSize      = (20, 20)
            )
            if len(faces) > 0:
                # Ambil wajah terbesar
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]

                # Tambah padding 20%
                pad = int(0.20 * min(w, h))
                x1  = max(0, x - pad)
                y1  = max(0, y - pad)
                x2  = min(image.shape[1], x + w + pad)
                y2  = min(image.shape[0], y + h + pad)

                face_crop = image[y1:y2, x1:x2]
                return cv2.resize(face_crop, FACE_SIZE)

    # Tidak ada wajah terdeteksi
    return None


# ─────────────────────────────────────────────
# 3. AUGMENTASI
# ─────────────────────────────────────────────
def augment_image(image: np.ndarray) -> list:
    augmented = []

    # 1. Flip horizontal
    augmented.append(cv2.flip(image, 1))

    # 2. Brightness terang
    augmented.append(cv2.convertScaleAbs(image, alpha=1.3, beta=30))

    # 3. Brightness gelap
    augmented.append(cv2.convertScaleAbs(image, alpha=0.7, beta=-20))

    # 4. Rotasi kiri 10°
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
    augmented.append(cv2.warpAffine(image, M, (w, h)))

    # 5. Rotasi kanan 10°
    M = cv2.getRotationMatrix2D((w//2, h//2), -10, 1.0)
    augmented.append(cv2.warpAffine(image, M, (w, h)))

    # 6. Gaussian noise
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    # 7. Contrast tinggi
    augmented.append(cv2.convertScaleAbs(image, alpha=1.5, beta=0))

    # 8. Blur ringan (simulasi kamera blur)
    augmented.append(cv2.GaussianBlur(image, (3, 3), 0))

    return augmented[:AUGMENT_PER_IMAGE]


# ─────────────────────────────────────────────
# 4. PIPELINE UTAMA
# ─────────────────────────────────────────────
def preprocess_dataset():
    if not os.path.exists(DATASET_RAW):
        logger.error(f"Folder tidak ditemukan: {DATASET_RAW}")
        return

    persons = [d for d in os.listdir(DATASET_RAW)
               if os.path.isdir(os.path.join(DATASET_RAW, d))]

    if not persons:
        logger.error("Tidak ada subfolder orang di dataset/raw/")
        return

    logger.info(f"Ditemukan {len(persons)} orang: {persons}")

    total_success = 0
    total_fail    = 0

    for person in persons:
        raw_dir  = os.path.join(DATASET_RAW, person)
        proc_dir = os.path.join(DATASET_PROC, person)
        os.makedirs(proc_dir, exist_ok=True)

        photo_files = [f for f in os.listdir(raw_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(photo_files) < MIN_PHOTOS_PER_PERSON:
            logger.warning(f"[{person}] hanya {len(photo_files)} foto (min: {MIN_PHOTOS_PER_PERSON})")

        logger.info(f"[{person}] Memproses {len(photo_files)} foto ...")
        saved  = 0
        failed = 0

        for idx, fname in enumerate(tqdm(photo_files, desc=f"  {person}")):
            fpath = os.path.join(raw_dir, fname)
            img   = cv2.imread(fpath)

            if img is None:
                failed += 1
                continue

            enhanced = apply_clahe(img)
            face     = detect_and_crop_face(enhanced)

            if face is None:
                failed += 1
                continue

            # Simpan foto asli
            cv2.imwrite(os.path.join(proc_dir, f"{idx:04d}_orig.jpg"), face)
            saved += 1

            # Augmentasi
            for aug_idx, aug_img in enumerate(augment_image(face)):
                cv2.imwrite(os.path.join(proc_dir, f"{idx:04d}_aug{aug_idx}.jpg"), aug_img)
                saved += 1

        logger.info(f"  [{person}] Tersimpan: {saved} | Gagal: {failed}")
        total_success += saved
        total_fail    += failed

    logger.info(f"PREPROCESSING SELESAI")
    logger.info(f"Total berhasil : {total_success}")
    logger.info(f"Total gagal    : {total_fail}")


# ─────────────────────────────────────────────
# PREPROCESSING REALTIME (dipanggil predict.py)
# ─────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray | None:
    if frame is None:
        return None
    enhanced = apply_clahe(frame)
    return detect_and_crop_face(enhanced)


if __name__ == "__main__":
    logger.info("Mulai preprocessing dataset Boostify ...")
    preprocess_dataset()