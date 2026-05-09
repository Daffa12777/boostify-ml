"""
preprocess.py — Preprocessing Dataset Boostify
================================================
Fungsi:
  1. Baca foto mentah dari dataset/raw/[nama]/
  2. Terapkan CLAHE (perbaiki pencahayaan gelap)
  3. Augmentasi (flip, brightness, rotation)
  4. Simpan hasil ke dataset/processed/[nama]/

Cara pakai:
  python preprocess.py

Output:
  dataset/processed/ berisi foto yang sudah bersih & diperbanyak
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
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
    """
    Perbaiki pencahayaan gambar pakai CLAHE (Contrast Limited Adaptive 
    Histogram Equalization). Efektif untuk ruangan gelap / backlight.
    
    Input : gambar BGR (dari OpenCV)
    Output: gambar BGR yang sudah diperbaiki pencahayaannya
    """
    # Convert ke LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Terapkan CLAHE hanya pada channel L (lightness)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_enhanced = clahe.apply(l)

    # Gabungkan kembali
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# 2. DETEKSI & CROP WAJAH
# ─────────────────────────────────────────────
def detect_and_crop_face(image: np.ndarray) -> np.ndarray | None:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Coba dari parameter paling longgar dulu
    for scale in [1.05, 1.1, 1.2]:
        for neighbors in [2, 3, 5]:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=(20, 20)   # lebih kecil dari sebelumnya
            )
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                pad = int(0.20 * min(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                face_crop = image[y1:y2, x1:x2]
                return cv2.resize(face_crop, FACE_SIZE)
    
    return None

    # Ambil wajah terbesar
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Tambah padding 20% di setiap sisi
    pad = int(0.20 * min(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    face_crop = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, FACE_SIZE)
    return face_resized


# ─────────────────────────────────────────────
# 3. AUGMENTASI
# ─────────────────────────────────────────────
def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """
    Buat variasi dari 1 foto untuk memperbanyak data training.
    Menghasilkan N variasi sesuai AUGMENT_PER_IMAGE di config.
    
    Variasi: flip horizontal, brightness +/-, rotasi kecil, noise
    """
    augmented = []

    # 1. Flip horizontal (cermin)
    augmented.append(cv2.flip(image, 1))

    # 2. Brightness lebih terang
    bright = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    augmented.append(bright)

    # 3. Brightness lebih gelap (simulasi ruang gelap)
    dark = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)
    augmented.append(dark)

    # 4. Rotasi kiri 10°
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
    rotated_left = cv2.warpAffine(image, M, (w, h))
    augmented.append(rotated_left)

    # 5. Rotasi kanan 10°
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -10, 1.0)
    rotated_right = cv2.warpAffine(image, M, (w, h))
    augmented.append(rotated_right)

    # 6. Tambah Gaussian noise (simulasi kamera jelek)
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    return augmented[:AUGMENT_PER_IMAGE]


# ─────────────────────────────────────────────
# 4. PIPELINE UTAMA
# ─────────────────────────────────────────────
def preprocess_dataset():
    """
    Baca semua foto dari dataset/raw/[nama]/,
    terapkan CLAHE + crop wajah + augmentasi,
    simpan ke dataset/processed/[nama]/.
    """
    if not os.path.exists(DATASET_RAW):
        logger.error(f"Folder dataset tidak ditemukan: {DATASET_RAW}")
        logger.info("Buat folder: ml/dataset/raw/[nama_orang]/ lalu isi dengan foto.")
        return

    persons = [d for d in os.listdir(DATASET_RAW)
               if os.path.isdir(os.path.join(DATASET_RAW, d))]

    if len(persons) == 0:
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
            logger.warning(
                f"[{person}] hanya {len(photo_files)} foto "
                f"(min: {MIN_PHOTOS_PER_PERSON}). Disarankan tambah foto."
            )

        logger.info(f"[{person}] Memproses {len(photo_files)} foto ...")
        saved = 0
        failed = 0

        for idx, fname in enumerate(tqdm(photo_files, desc=f"  {person}")):
            fpath  = os.path.join(raw_dir, fname)
            img    = cv2.imread(fpath)

            if img is None:
                logger.warning(f"  Gagal baca: {fname}")
                failed += 1
                continue

            # Step 1: CLAHE
            enhanced = apply_clahe(img)

            # Step 2: Crop wajah
            face = detect_and_crop_face(enhanced)
            if face is None:
                logger.warning(f"  Wajah tidak terdeteksi: {fname}")
                failed += 1
                continue

            # Step 3: Simpan foto asli yang sudah diproses
            out_path = os.path.join(proc_dir, f"{idx:04d}_orig.jpg")
            cv2.imwrite(out_path, face)
            saved += 1

            # Step 4: Augmentasi
            augmented = augment_image(face)
            for aug_idx, aug_img in enumerate(augmented):
                aug_path = os.path.join(proc_dir, f"{idx:04d}_aug{aug_idx}.jpg")
                cv2.imwrite(aug_path, aug_img)
                saved += 1

        logger.info(f"  [{person}] ✅ Tersimpan: {saved} | ❌ Gagal: {failed}")
        total_success += saved
        total_fail    += failed

    logger.info(f"\n{'='*50}")
    logger.info(f"PREPROCESSING SELESAI")
    logger.info(f"Total foto berhasil diproses : {total_success}")
    logger.info(f"Total foto gagal             : {total_fail}")
    logger.info(f"Output di: {DATASET_PROC}")


# ─────────────────────────────────────────────
# SINGLE IMAGE PREPROCESSING (untuk realtime)
# ─────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray | None:
    """
    Preprocessing satu frame dari kamera secara realtime.
    Dipanggil oleh predict.py saat absensi berlangsung.
    
    Input : frame BGR dari OpenCV
    Output: wajah yang sudah di-crop & enhance, atau None
    """
    if frame is None:
        return None

    enhanced = apply_clahe(frame)
    face     = detect_and_crop_face(enhanced)
    return face


if __name__ == "__main__":
    logger.info("🚀 Mulai preprocessing dataset Boostify ...")
    preprocess_dataset()
