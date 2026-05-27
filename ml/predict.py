"""
predict.py — Face Recognition Engine Boostify
Model: dlib ResNet (via library face_recognition, 128-dim)
+ Smile Detection: Simple MAR (Mouth Aspect Ratio)
  → Paling ringan, tanpa model tambahan
  → Tidak tambah beban CPU Raspberry Pi

Interface (recognize → dict) DIBUAT SAMA PERSIS dengan versi lama,
jadi predict_thread.py & LCD tidak perlu diubah.
"""

import os
import sys
import pickle
import time
import uuid
import numpy as np
import cv2
import face_recognition
from datetime import datetime, timezone
from typing import Optional
from collections import deque
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    EMBEDDING_FILE, LABEL_FILE,
    SIMILARITY_THRESHOLD,
    CLAHE_CLIP, CLAHE_GRID,
    COOLDOWN_SEC,
    CAMERA_INDEX, FRAME_SKIP,
    SMILE_VOTE_FRAMES, SMILE_VOTE_THRESH,
    FR_DETECTOR, FR_NUM_JITTERS
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
# KODE CUSTOM PER ORANG — EDIT DI SINI
# ─────────────────────────────────────────────
KODE_ASISTEN = {
    "daffa"   : "FDR",
    "dirgi"   : "DRG",
    "rufus"   : "RFS",
    "DAFFAFR" : "FDL",
    # "nama"  : "KODE",
}

def generate_code(nama: str) -> str:
    return KODE_ASISTEN.get(nama.lower(), nama[:3].upper())


# ─────────────────────────────────────────────
# FORMAT WAKTU
# ─────────────────────────────────────────────
def get_time_data() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "time"         : now.isoformat(),
        "formattedTime": now.strftime("%A, %B %d, %Y")
    }


# ─────────────────────────────────────────────
# SMILE DETECTION — Simple MAR  (TIDAK DIUBAH)
# ─────────────────────────────────────────────
_smile_votes = deque(maxlen=SMILE_VOTE_FRAMES)
SMILE_WHITE_THRESHOLD = 0.06


def _deteksi_senyum_mar(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    y1 = int(h * 0.60)
    y2 = int(h * 0.85)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    mouth_roi = gray[y1:y2, x1:x2]
    if mouth_roi.size == 0:
        return False

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    mouth_eq = clahe.apply(mouth_roi)

    thresh = cv2.adaptiveThreshold(
        mouth_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, -5
    )

    white_ratio = np.sum(thresh == 255) / thresh.size
    return white_ratio > SMILE_WHITE_THRESHOLD


def detect_smile(frame: np.ndarray) -> bool:
    hasil = _deteksi_senyum_mar(frame)
    _smile_votes.append(hasil)

    jumlah_senyum = sum(_smile_votes)
    if len(_smile_votes) >= SMILE_VOTE_FRAMES:
        return jumlah_senyum >= SMILE_VOTE_THRESH
    return hasil


def reset_smile_buffer():
    _smile_votes.clear()


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
# CLASS FACE RECOGNIZER (dlib ResNet)
# ─────────────────────────────────────────────
class FaceRecognizer:

    def __init__(self):
        logger.info("Inisialisasi FaceRecognizer (dlib ResNet / face_recognition) ...")
        self.embeddings_db       = {}
        self.labels              = []
        self._last_detected_time = {}
        self._load_database()
        logger.info(f"Siap. {len(self.labels)} orang terdaftar: {self.labels}")

    def _load_database(self):
        if not os.path.exists(EMBEDDING_FILE):
            raise FileNotFoundError("Jalankan train.py dulu!")
        with open(EMBEDDING_FILE, "rb") as f:
            self.embeddings_db = pickle.load(f)
        with open(LABEL_FILE, "rb") as f:
            self.labels = pickle.load(f)

    # ─────────────────────────────────────────
    # DETECT + EXTRACT ENCODING dari frame kamera
    # ─────────────────────────────────────────
    def _detect_and_encode(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Return encoding 128-d dari wajah TERBESAR di frame, atau None.
        """
        enhanced = apply_clahe(frame_bgr)
        rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model=FR_DETECTOR)
        if not locations:
            return None

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

        return encodings[0]

    # ─────────────────────────────────────────
    # FIND BEST MATCH — Euclidean distance (native dlib)
    # confidence = 1 - distance (clamped to 0..1)
    # ─────────────────────────────────────────
    def _find_best_match(self, emb: np.ndarray) -> tuple:
        best_name     = "Unknown"
        best_distance = float("inf")

        for nama, db_emb in self.embeddings_db.items():
            d = float(np.linalg.norm(emb - db_emb))
            if d < best_distance:
                best_distance = d
                best_name     = nama

        confidence = max(0.0, 1.0 - best_distance)
        return best_name, confidence

    def _on_cooldown(self, nama: str) -> bool:
        return (time.time() - self._last_detected_time.get(nama, 0)) < COOLDOWN_SEC

    # ─────────────────────────────────────────
    # FUNGSI UTAMA — dipanggil IoT/predict_thread
    # Output dict DIBUAT SAMA PERSIS dengan versi lama.
    # ─────────────────────────────────────────
    def recognize(self, frame: np.ndarray) -> dict:
        # Step 1: deteksi + ekstrak encoding
        emb = self._detect_and_encode(frame)
        if emb is None:
            detect_smile(frame)
            return {
                "status": "no_face", "assisstant_code": "",
                "name": "", "confidence": 0.0, "time": "",
                "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": ""
            }

        # Step 2: cari kecocokan
        nama, confidence = self._find_best_match(emb)

        # Step 3: cek threshold
        if confidence < SIMILARITY_THRESHOLD:
            detect_smile(frame)
            return {
                "status": "unknown", "assisstant_code": "",
                "name": "Unknown", "confidence": round(confidence, 3),
                "time": "", "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": "Wajah tidak dikenal"
            }

        # Step 4: cek cooldown
        if self._on_cooldown(nama):
            detect_smile(frame)
            return {
                "status": "cooldown", "assisstant_code": generate_code(nama),
                "name": nama, "confidence": round(confidence, 3),
                "time": "", "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": f"Hai, {nama}! Sudah absen."
            }

        # Step 5: sukses
        self._last_detected_time[nama] = time.time()
        time_data  = get_time_data()
        is_smiling = bool(detect_smile(frame))
        reset_smile_buffer()

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
# TEST REALTIME (kalau dijalanin langsung)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logger.info("Test predict.py — dlib ResNet + Simple MAR Smile Detection")
    logger.info(f"Detector: {FR_DETECTOR} | Threshold: {SIMILARITY_THRESHOLD}")

    recognizer       = FaceRecognizer()
    cap              = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_count      = 0
    last_result      = {}
    last_upload_time = {}

    time.sleep(1)
    logger.info("Kamera aktif. Tekan Q untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            last_result  = recognizer.recognize(frame)
            nama         = last_result.get("name", "")
            current_time = time.time()

            if last_result["status"] == "recognized":
                if nama not in last_upload_time:
                    last_upload_time[nama] = 0

                if current_time - last_upload_time[nama] > COOLDOWN_SEC:
                    print("\n" + "=" * 50)
                    print("DATA ABSENSI:")
                    print(json.dumps({
                        "assisstant_code": last_result["assisstant_code"],
                        "name"           : last_result["name"],
                        "time"           : last_result["time"],
                        "uuid"           : last_result["uuid"],
                        "formattedTime"  : last_result["formattedTime"],
                        "is_smiling"     : last_result["is_smiling"]
                    }, indent=4))
                    print("=" * 50)

                    try:
                        response = requests.post(
                            "http://localhost:3000/api/uploadfromml",
                            json=last_result
                        )
                        print("API RESPONSE:", response.text)
                        last_upload_time[nama] = current_time
                    except Exception as e:
                        print("GAGAL KIRIM KE BACKEND:", e)

        # ── Tampilan kamera ──
        display = frame.copy()

        # Visualisasi area mulut yang dianalisis (smile MAR)
        h, w = display.shape[:2]
        y1m = int(h * 0.60)
        y2m = int(h * 0.85)
        x1m = int(w * 0.25)
        x2m = int(w * 0.75)
        cv2.rectangle(display, (x1m, y1m), (x2m, y2m), (0, 255, 255), 1)

        if last_result:
            status     = last_result.get("status", "")
            nama       = last_result.get("name", "")
            conf       = last_result.get("confidence", 0.0)
            code       = last_result.get("assisstant_code", "")
            is_smiling = last_result.get("is_smiling", False)

            color = (
                (0, 200, 0)   if status == "recognized" else
                (0, 0, 220)   if status == "unknown"    else
                (180, 180, 0) if status == "cooldown"   else
                (100, 100, 100)
            )

            cv2.putText(display, f"{nama} [{code}]",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, f"conf: {conf:.2f} | {status}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if status == "recognized":
                vote_count = sum(_smile_votes)
                vote_total = len(_smile_votes)
                smile_text = f"Senyum: {'YES :)' if is_smiling else 'NO'} ({vote_count}/{vote_total})"
                cv2.putText(display, smile_text,
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 220, 255), 2)
                cv2.putText(display, last_result.get("formattedTime", ""),
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)

        cv2.putText(display, "Model: dlib ResNet | Smile: Simple MAR",
                    (20, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 200, 100), 1)

        cv2.imshow("Boostify — dlib ResNet + MAR Smile", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()