"""
predict.py — Face Recognition Engine Boostify
Output format disesuaikan dengan API /api/attendances
+ Smile Detection (OpenCV Haar Cascade)
"""

import os
import sys
import pickle
import time
import uuid
import tempfile
import numpy as np
import cv2
from datetime import datetime, timezone
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

TMP_DETECT = os.path.join(tempfile.gettempdir(), "boostify_detect.jpg")
TMP_FACE   = os.path.join(tempfile.gettempdir(), "boostify_face.jpg")

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
# KODE CUSTOM PER ORANG — EDIT DI SINI
# ─────────────────────────────────────────────
KODE_ASISTEN = {
    "daffa" : "FDR",
    "dirgi" : "DRG",
    "dea"   : "DEA",
    # tambah orang baru:
    # "nama" : "KODE",
}

def generate_code(nama: str) -> str:
    return KODE_ASISTEN.get(nama.lower(), nama[:3].upper())


# ─────────────────────────────────────────────
# FORMAT WAKTU
# ─────────────────────────────────────────────
def get_time_data() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "time"         : now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z",
        "formattedTime": now.strftime("%A, %B %d, %Y")
    }


# ─────────────────────────────────────────────
# SMILE DETECTION — Ringan, pakai OpenCV
# ─────────────────────────────────────────────
smile_cascade      = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
face_cascade_smile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_smile(frame: np.ndarray) -> bool:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_smile.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        roi    = gray[y + h//2 : y + h, x : x + w]
        smiles = smile_cascade.detectMultiScale(
            roi, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25)
        )
        if len(smiles) > 0:
            return True
    return False


# ─────────────────────────────────────────────
# CLAHE ENHANCEMENT
# ─────────────────────────────────────────────
def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# CROP WAJAH
# ─────────────────────────────────────────────
def get_face_crop(image: np.ndarray, DeepFace) -> np.ndarray:
    try:
        cv2.imwrite(TMP_DETECT, image)
        faces = DeepFace.extract_faces(
            img_path=TMP_DETECT, detector_backend="opencv",
            enforce_detection=False, align=True
        )
        if faces and len(faces) > 0:
            face_data = faces[0]["face"]
            if face_data.max() <= 1.0:
                face_data = (face_data * 255).astype(np.uint8)
            return cv2.resize(face_data, FACE_SIZE)
    except Exception as e:
        logger.debug(f"DeepFace detect gagal: {e}")

    h, w = image.shape[:2]
    size = min(h, w)
    y1   = (h - size) // 2
    x1   = (w - size) // 2
    return cv2.resize(image[y1:y1+size, x1:x1+size], FACE_SIZE)


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
            raise FileNotFoundError("Jalankan train.py dulu!")
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
            cv2.imwrite(TMP_FACE, face_img)
            result = self.deepface.represent(
                img_path=TMP_FACE, model_name=MODEL_BACKEND,
                detector_backend="skip", enforce_detection=False, align=False
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
        # Step 1: enhance + crop
        enhanced = apply_clahe(frame)
        face     = get_face_crop(enhanced, self.deepface)

        # Step 2: extract embedding
        emb = self._extract_embedding(face)
        if emb is None:
            return {
                "status": "no_face", "assisstant_code": "",
                "name": "", "confidence": 0.0, "time": "",
                "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": ""
            }

        # Step 3: cari kecocokan
        nama, confidence = self._find_best_match(emb)

        # Step 4: cek threshold
        if confidence < SIMILARITY_THRESHOLD:
            return {
                "status": "unknown", "assisstant_code": "",
                "name": "Unknown", "confidence": round(confidence, 3),
                "time": "", "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": "Wajah tidak dikenal"
            }

        # Step 5: cek cooldown
        if self._on_cooldown(nama):
            return {
                "status": "cooldown", "assisstant_code": generate_code(nama),
                "name": nama, "confidence": round(confidence, 3),
                "time": "", "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": f"Hai, {nama}! Sudah absen."
            }

        # Step 6: sukses
        self._last_detected_time[nama] = time.time()
        time_data  = get_time_data()
        is_smiling = detect_smile(frame)

        attendance = {
            "status"          : "recognized",
            "assisstant_code" : generate_code(nama),
            "name"            : nama,
            "confidence"      : round(confidence, 3),
            "time"            : time_data["time"],
            "uuid"            : str(uuid.uuid4()),
            "formattedTime"   : time_data["formattedTime"],
            "is_smiling"      : is_smiling,
            "message"         : f"Selamat Datang, {nama}!\n{'Senyumnya Keren!' if is_smiling else 'Semangat!'}"
        }

        logger.info(
            f"ABSEN: {nama} | code: {attendance['assisstant_code']} | "
            f"confidence: {confidence:.3f} | senyum: {is_smiling} | "
            f"uuid: {attendance['uuid']}"
        )

        return attendance

    def reload_database(self):
        self._load_database()
        logger.info(f"Database diperbarui: {self.labels}")


# ─────────────────────────────────────────────
# TEST REALTIME
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logger.info("Test predict.py — Realtime Recognition + Smile Detection")
    recognizer  = FaceRecognizer()
    cap         = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    frame_count = 0
    last_result = {}

    time.sleep(1)
    logger.info("Kamera aktif. Tekan Q untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            last_result = recognizer.recognize(frame)

            if last_result["status"] == "recognized":
                print("\n" + "="*50)
                print("DATA ABSENSI (siap kirim ke API):")
                print(json.dumps({
                    "assisstant_code": last_result["assisstant_code"],
                    "name"           : last_result["name"],
                    "time"           : last_result["time"],
                    "uuid"           : last_result["uuid"],
                    "formattedTime"  : last_result["formattedTime"],
                    "is_smiling"     : last_result["is_smiling"]
                }, indent=4))
                print("="*50)

        # Tampilan kamera
        display = frame.copy()
        if last_result:
            status     = last_result.get("status", "")
            nama       = last_result.get("name", "")
            conf       = last_result.get("confidence", 0.0)
            code       = last_result.get("assisstant_code", "")
            is_smiling = last_result.get("is_smiling", False)

            color = (0, 200, 0)   if status == "recognized" else \
                    (0, 0, 220)   if status == "unknown"    else \
                    (180, 180, 0) if status == "cooldown"   else \
                    (100, 100, 100)

            cv2.putText(display, f"{nama} [{code}]",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, f"conf: {conf:.2f} | {status}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if status == "recognized":
                smile_text = "Senyum: YES :)" if is_smiling else "Senyum: NO"
                cv2.putText(display, smile_text,
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                cv2.putText(display, last_result.get("formattedTime", ""),
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Boostify — Recognition + Smile", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()