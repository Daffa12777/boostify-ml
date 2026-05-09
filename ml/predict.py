"""
predict.py — Face Recognition Engine Boostify (Fixed)
Menggunakan DeepFace built-in detector yang lebih robust
dari Haar Cascade.
"""

import os
import sys
import pickle
import time
import numpy as np
import cv2
import tempfile
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    EMBEDDING_FILE, LABEL_FILE,
    SIMILARITY_THRESHOLD,
    MODEL_BACKEND,
    FACE_SIZE,
    CLAHE_CLIP, CLAHE_GRID,
    COOLDOWN_SEC,
    CAMERA_INDEX, FRAME_SKIP
)
from utils.logger import get_logger

logger = get_logger("predict")

MOTIVASI_MESSAGES = [
    "Semangat Hari Ini!",
    "Kamu Bisa!",
    "Jangan Menyerah!",
    "Selamat Bekerja!",
    "Ayo Produktif!",
    "Tetap Semangat!",
    "Luar Biasa!",
    "Sukses Selalu!"
]

def get_random_message() -> str:
    import random
    return random.choice(MOTIVASI_MESSAGES)


# ─────────────────────────────────────────────
# CLAHE ENHANCEMENT
# ─────────────────────────────────────────────
def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# CROP WAJAH — pakai DeepFace detector
# Kalau gagal → pakai center crop (fallback)
# ─────────────────────────────────────────────
def get_face_crop(image: np.ndarray, DeepFace) -> np.ndarray:
    """
    Coba deteksi wajah pakai DeepFace (lebih akurat dari Haar Cascade).
    Kalau tidak terdeteksi → pakai center crop sebagai fallback.
    Sistem absensi = orang berdiri di depan kamera → center crop cukup.
    """
    try:
        # Simpan frame sementara
        tmp = os.path.join(tempfile.gettempdir(), "boostify_detect.jpg")
        cv2.imwrite(tmp, image)

        faces = DeepFace.extract_faces(
            img_path          = tmp,
            detector_backend  = "opencv",
            enforce_detection = False,
            align             = True
        )

        if faces and len(faces) > 0:
            face_data = faces[0]["face"]
            # DeepFace return float 0-1, convert ke uint8
            if face_data.max() <= 1.0:
                face_data = (face_data * 255).astype(np.uint8)
            return cv2.resize(face_data, FACE_SIZE)

    except Exception as e:
        logger.debug(f"DeepFace detect gagal: {e}")

    # ── Fallback: center crop ──
    h, w = image.shape[:2]
    size  = min(h, w)
    y1    = (h - size) // 2
    x1    = (w - size) // 2
    crop  = image[y1:y1+size, x1:x1+size]
    return cv2.resize(crop, FACE_SIZE)


# ─────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ─────────────────────────────────────────────
# CLASS FACE RECOGNIZER
# ─────────────────────────────────────────────
class FaceRecognizer:

    def __init__(self):
        logger.info("Inisialisasi FaceRecognizer ...")
        self.embeddings_db       = {}
        self.labels              = []
        self.deepface            = None
        self._last_detected_time = {}
        self._load_database()
        self._load_model()
        logger.info(f"Siap. {len(self.labels)} orang terdaftar: {self.labels}")

    def _load_database(self):
        if not os.path.exists(EMBEDDING_FILE):
            raise FileNotFoundError(f"Jalankan train.py dulu!")
        with open(EMBEDDING_FILE, "rb") as f:
            self.embeddings_db = pickle.load(f)
        with open(LABEL_FILE, "rb") as f:
            self.labels = pickle.load(f)

    def _load_model(self):
        from deepface import DeepFace
        self.deepface = DeepFace
        logger.info(f"Model {MODEL_BACKEND} dimuat.")

    def _extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        try:
            tmp = os.path.join(tempfile.gettempdir(), "boostify_face.jpg")
            cv2.imwrite(tmp, face_img)
            result = self.deepface.represent(
                img_path          = tmp,
                model_name        = MODEL_BACKEND,
                detector_backend  = "skip",
                enforce_detection = False,
                align             = False
            )
            if result:
                emb  = np.array(result[0]["embedding"])
                norm = np.linalg.norm(emb)
                return emb / norm if norm > 0 else emb
        except Exception as e:
            logger.debug(f"Extract embedding gagal: {e}")
        return None

    def _find_best_match(self, emb: np.ndarray) -> tuple:
        best_name, best_score = "Unknown", 0.0
        for nama, db_emb in self.embeddings_db.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                best_name  = nama
        return best_name, best_score

    def _on_cooldown(self, nama: str) -> bool:
        return (time.time() - self._last_detected_time.get(nama, 0)) < COOLDOWN_SEC

    def recognize(self, frame: np.ndarray) -> dict:
        # Step 1: enhance
        enhanced = apply_clahe(frame)

        # Step 2: crop wajah
        face = get_face_crop(enhanced, self.deepface)

        # Step 3: extract embedding
        emb = self._extract_embedding(face)
        if emb is None:
            return {"status": "no_face", "nama": "", "confidence": 0.0, "message": ""}

        # Step 4: cari kecocokan
        nama, confidence = self._find_best_match(emb)

        # Step 5: cek threshold
        if confidence < SIMILARITY_THRESHOLD:
            return {
                "status"    : "unknown",
                "nama"      : "Unknown",
                "confidence": round(confidence, 3),
                "message"   : "Wajah tidak dikenal"
            }

        # Step 6: cek cooldown
        if self._on_cooldown(nama):
            return {
                "status"    : "cooldown",
                "nama"      : nama,
                "confidence": round(confidence, 3),
                "message"   : f"Hai, {nama}!"
            }

        # Step 7: sukses!
        self._last_detected_time[nama] = time.time()
        logger.info(f"ABSEN: {nama} | confidence: {confidence:.3f}")

        return {
            "status"    : "recognized",
            "nama"      : nama,
            "confidence": round(confidence, 3),
            "message"   : f"Selamat Datang, {nama}!\n{get_random_message()}"
        }

    def reload_database(self):
        self._load_database()
        logger.info(f"Database diperbarui: {self.labels}")


# ─────────────────────────────────────────────
# TEST REALTIME
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Test predict.py — Realtime Recognition")
    recognizer  = FaceRecognizer()
    cap         = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    frame_count = 0
    last_result = {}

    # Warmup kamera
    time.sleep(1)
    logger.info("Kamera aktif. Tekan Q untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Gagal baca frame kamera!")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            last_result = recognizer.recognize(frame)

        # ── Tampilan ──
        display = frame.copy()
        if last_result:
            status = last_result.get("status", "")
            nama   = last_result.get("nama", "")
            conf   = last_result.get("confidence", 0.0)

            color = (0, 200, 0)   if status == "recognized" else \
                    (0, 0, 220)   if status == "unknown"    else \
                    (180, 180, 0) if status == "cooldown"   else \
                    (100, 100, 100)

            cv2.putText(display, f"{nama}",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(display, f"conf: {conf:.2f} | {status}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if status == "recognized":
                msg = last_result.get("message", "").split("\n")
                if len(msg) > 1:
                    cv2.putText(display, msg[1],
                                (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 220, 100), 2)

        cv2.imshow("Boostify — Test Recognition", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()