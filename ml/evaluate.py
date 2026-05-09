"""
evaluate.py — Evaluasi Akurasi Model Boostify (Fixed for Windows)
"""

import os
import sys
import pickle
import tempfile
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_DIR, EMBEDDING_FILE, LABEL_FILE,
    SIMILARITY_THRESHOLD, MODEL_BACKEND,
    FACE_SIZE, CLAHE_CLIP, CLAHE_GRID
)
from utils.logger import get_logger

logger    = get_logger("evaluate")
TMP_EVAL  = os.path.join(tempfile.gettempdir(), "boostify_eval.jpg")
TMP_FACE  = os.path.join(tempfile.gettempdir(), "boostify_eval_face.jpg")

DATASET_TEST = os.path.join(BASE_DIR, "dataset", "test")


def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def get_face_crop(image: np.ndarray, DeepFace) -> np.ndarray:
    """Crop wajah, fallback ke center crop kalau gagal."""
    try:
        cv2.imwrite(TMP_EVAL, image)
        faces = DeepFace.extract_faces(
            img_path          = TMP_EVAL,
            detector_backend  = "opencv",
            enforce_detection = False,
            align             = True
        )
        if faces and len(faces) > 0:
            face_data = faces[0]["face"]
            if face_data.max() <= 1.0:
                face_data = (face_data * 255).astype(np.uint8)
            return cv2.resize(face_data, FACE_SIZE)
    except Exception as e:
        logger.debug(f"Detect gagal: {e}")

    # Fallback: center crop
    h, w = image.shape[:2]
    size = min(h, w)
    y1   = (h - size) // 2
    x1   = (w - size) // 2
    return cv2.resize(image[y1:y1+size, x1:x1+size], FACE_SIZE)


def extract_embedding(DeepFace, image_path: str) -> np.ndarray | None:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        enhanced = apply_clahe(img)
        face     = get_face_crop(enhanced, DeepFace)

        cv2.imwrite(TMP_FACE, face)
        result = DeepFace.represent(
            img_path          = TMP_FACE,
            model_name        = MODEL_BACKEND,
            detector_backend  = "skip",
            enforce_detection = False,
            align             = False
        )
        if result and len(result) > 0:
            emb  = np.array(result[0]["embedding"])
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        return None
    except Exception as e:
        logger.debug(f"Gagal: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def find_best_match(emb: np.ndarray, db: dict) -> tuple:
    best_name, best_score = "Unknown", 0.0
    for nama, db_emb in db.items():
        score = cosine_similarity(emb, db_emb)
        if score > best_score:
            best_score = score
            best_name  = nama
    return best_name, best_score


def find_optimal_threshold(results_raw: list) -> float:
    best_thresh, best_acc = 0.5, 0.0
    for thresh in np.arange(0.3, 0.95, 0.05):
        correct = sum(
            1 for true_l, pred_l, score in results_raw
            if (pred_l if score >= thresh else "Unknown") == true_l
        )
        acc = correct / len(results_raw) if results_raw else 0
        if acc > best_acc:
            best_acc   = acc
            best_thresh = thresh
    return round(best_thresh, 2)


def evaluate():
    if not os.path.exists(EMBEDDING_FILE):
        logger.error("embeddings.pkl tidak ditemukan. Jalankan train.py dulu.")
        return

    if not os.path.exists(DATASET_TEST):
        logger.error(f"Folder test tidak ditemukan: {DATASET_TEST}")
        logger.info("Buat: ml/dataset/test/[nama]/ lalu isi foto test.")
        return

    with open(EMBEDDING_FILE, "rb") as f:
        embeddings_db = pickle.load(f)
    with open(LABEL_FILE, "rb") as f:
        registered_labels = pickle.load(f)

    logger.info(f"Database: {registered_labels}")

    try:
        from deepface import DeepFace
    except ImportError:
        logger.error("pip install deepface")
        return

    persons = [d for d in os.listdir(DATASET_TEST)
               if os.path.isdir(os.path.join(DATASET_TEST, d))]

    if not persons:
        logger.error("Tidak ada subfolder di dataset/test/")
        return

    logger.info(f"Evaluasi untuk {len(persons)} orang: {persons}\n")

    results_raw = []
    per_person  = defaultdict(lambda: {"correct": 0, "wrong": 0, "no_face": 0})

    for person in persons:
        test_dir    = os.path.join(DATASET_TEST, person)
        test_photos = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not test_photos:
            logger.warning(f"[{person}] Tidak ada foto test.")
            continue

        logger.info(f"[{person}] Testing {len(test_photos)} foto ...")

        for fpath in tqdm(test_photos, desc=f"  {person}"):
            emb = extract_embedding(DeepFace, fpath)

            if emb is None:
                per_person[person]["no_face"] += 1
                results_raw.append((person, "Unknown", 0.0))
                continue

            pred_name, score = find_best_match(emb, embeddings_db)

            if score >= SIMILARITY_THRESHOLD and pred_name == person:
                per_person[person]["correct"] += 1
            else:
                per_person[person]["wrong"] += 1

            results_raw.append((person, pred_name, score))

    # ── Laporan ──
    print(f"\n{'='*60}")
    print("LAPORAN EVALUASI BOOSTIFY ML")
    print(f"{'='*60}")
    print(f"Threshold saat ini : {SIMILARITY_THRESHOLD}")
    print(f"Model              : {MODEL_BACKEND}")
    print(f"{'─'*60}")
    print(f"{'Nama':<20} {'Benar':>8} {'Salah':>8} {'No Face':>8} {'Akurasi':>10}")
    print(f"{'─'*60}")

    total_correct = total_wrong = total_noface = 0

    for person in persons:
        c  = per_person[person]["correct"]
        w  = per_person[person]["wrong"]
        nf = per_person[person]["no_face"]
        total_test = c + w + nf
        acc = f"{(c/total_test*100):.1f}%" if total_test > 0 else "N/A"
        print(f"{person:<20} {c:>8} {w:>8} {nf:>8} {acc:>10}")
        total_correct += c
        total_wrong   += w
        total_noface  += nf

    total_all = total_correct + total_wrong + total_noface
    overall   = total_correct / total_all * 100 if total_all > 0 else 0

    print(f"{'─'*60}")
    print(f"{'TOTAL':<20} {total_correct:>8} {total_wrong:>8} {total_noface:>8} {overall:>9.1f}%")
    print(f"{'='*60}")

    optimal = find_optimal_threshold(results_raw)
    print(f"\nREKOMENDASI:")
    print(f"  Threshold optimal: {optimal}")
    if optimal != SIMILARITY_THRESHOLD:
        print(f"  Ubah SIMILARITY_THRESHOLD di config.py ke {optimal}")

    if overall >= 90:
        print(f"\n  Akurasi {overall:.1f}% — Model SIAP deploy ke Raspberry Pi!")
    elif overall >= 75:
        print(f"\n  Akurasi {overall:.1f}% — Cukup, tapi tambah data untuk lebih baik.")
    else:
        print(f"\n  Akurasi {overall:.1f}% — Tambah dataset dulu sebelum deploy!")


if __name__ == "__main__":
    logger.info("Mulai evaluasi model Boostify ...")
    evaluate()