"""
predict.py — Face Recognition Engine Boostify
Model: dlib ResNet (via library face_recognition, 128-dim)
+ Smile Detection: LANDMARK-BASED (rasio lebar mulut : lebar wajah)
  → Diikat ke wajah asli (bukan region tetap di frame)
  → Tahan cahaya, kebaca senyum tertutup & terbuka
  → Tanpa model/lib tambahan (pakai 68-titik landmark dlib)

Interface (recognize → dict) tetap SAMA, jadi predict_thread.py & LCD aman.
"""

import os
import sys
import pickle
import time
import uuid
import platform
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
    FR_DETECTOR, FR_NUM_JITTERS,
    DETECT_SCALE
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
# SMILE DETECTION — LANDMARK BASED
# Rasio: lebar mulut / lebar wajah
#   → Senyum = mulut melebar = rasio naik
#   → Diikat ke wajah asli, tahan cahaya
# ─────────────────────────────────────────────

# Threshold rasio. Kalibrasi:
#   - Wajah netral biasanya ~0.33-0.40
#   - Senyum biasanya ~0.43+
# Naikkan kalau kebanyakan false-positive, turunkan kalau senyum nggak kedeteksi.
SMILE_RATIO_THRESHOLD = 0.42

_smile_votes      = deque(maxlen=SMILE_VOTE_FRAMES)
_last_smile_ratio = 0.0   # buat ditampilkan saat kalibrasi


def _euclidean(p1, p2) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _deteksi_senyum_landmark(rgb_full: np.ndarray, face_location) -> bool:
    """
    Deteksi senyum dari landmark mulut (dlib 68-point).
    face_location: (top, right, bottom, left) di koordinat full-res.
    """
    global _last_smile_ratio

    if face_location is None:
        return False

    lms = face_recognition.face_landmarks(rgb_full, face_locations=[face_location])
    if not lms:
        return False

    lm = lms[0]
    top_lip = lm.get("top_lip")
    if not top_lip or len(top_lip) < 7:
        return False

    # Sudut kiri & kanan mulut (titik ujung bibir atas)
    left_corner  = top_lip[0]
    right_corner = top_lip[6]
    mouth_width  = _euclidean(left_corner, right_corner)

    top, right, bottom, left = face_location
    face_width = max(1, right - left)

    ratio = mouth_width / face_width
    _last_smile_ratio = ratio
    return ratio > SMILE_RATIO_THRESHOLD


def detect_smile(rgb_full: np.ndarray, face_location) -> bool:
    """Voting multi-frame biar stabil & nggak flickering."""
    hasil = _deteksi_senyum_landmark(rgb_full, face_location)
    _smile_votes.append(hasil)

    if len(_smile_votes) >= SMILE_VOTE_FRAMES:
        return sum(_smile_votes) >= SMILE_VOTE_THRESH
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
    # DETECT + ENCODE
    # Deteksi di frame kecil (cepat), encode di full-res (akurat).
    # Return: (encoding_128d | None, face_location_full | None)
    # ─────────────────────────────────────────
    def _detect_and_encode(self, rgb_full: np.ndarray):
        # deteksi di frame yang diperkecil → HOG ngebut (penting di Pi)
        small = cv2.resize(rgb_full, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        locs  = face_recognition.face_locations(small, model=FR_DETECTOR)
        if not locs:
            return None, None

        # pilih wajah terbesar, lalu skalakan balik ke full-res
        locs = sorted(locs, key=lambda l: (l[2]-l[0]) * (l[1]-l[3]), reverse=True)
        t, r, b, l = locs[0]
        inv = 1.0 / DETECT_SCALE
        loc_full = (int(t*inv), int(r*inv), int(b*inv), int(l*inv))

        # encode di full-res biar kualitas encoding tetap bagus
        enc = face_recognition.face_encodings(
            rgb_full, known_face_locations=[loc_full], num_jitters=FR_NUM_JITTERS
        )
        if not enc:
            return None, loc_full
        return enc[0], loc_full

    # ─────────────────────────────────────────
    # FIND BEST MATCH — Euclidean distance (native dlib)
    # confidence = 1 - distance (clamped 0..1)
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
    # Output dict tetap SAMA.
    # ─────────────────────────────────────────
    def recognize(self, frame: np.ndarray) -> dict:
        # Enhance + convert sekali, dipakai untuk deteksi, encode, & smile
        enhanced = apply_clahe(frame)
        rgb_full = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        emb, loc = self._detect_and_encode(rgb_full)

        # Tidak ada wajah
        if emb is None:
            return {
                "status": "no_face", "assisstant_code": "",
                "name": "", "confidence": 0.0, "time": "",
                "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": ""
            }

        nama, confidence = self._find_best_match(emb)

        # Di bawah threshold → unknown
        if confidence < SIMILARITY_THRESHOLD:
            detect_smile(rgb_full, loc)   # update buffer voting
            return {
                "status": "unknown", "assisstant_code": "",
                "name": "Unknown", "confidence": round(confidence, 3),
                "time": "", "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": "Wajah tidak dikenal"
            }

        # Cooldown
        if self._on_cooldown(nama):
            detect_smile(rgb_full, loc)
            return {
                "status": "cooldown", "assisstant_code": generate_code(nama),
                "name": nama, "confidence": round(confidence, 3),
                "time": "", "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": f"Hai, {nama}! Sudah absen."
            }

        # Sukses
        self._last_detected_time[nama] = time.time()
        time_data  = get_time_data()
        is_smiling = bool(detect_smile(rgb_full, loc))
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
# Kamera lintas-OS: Windows pakai DSHOW, Linux/Pi pakai V4L2
# ─────────────────────────────────────────────
def open_camera(index: int):
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ─────────────────────────────────────────────
# TEST REALTIME (kalau dijalanin langsung)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logger.info("Test predict.py — dlib ResNet + Smile Landmark")
    logger.info(f"Detector: {FR_DETECTOR} | Threshold: {SIMILARITY_THRESHOLD} | SmileRatio: {SMILE_RATIO_THRESHOLD}")

    recognizer       = FaceRecognizer()
    cap              = open_camera(CAMERA_INDEX)
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

            # Tampilkan rasio senyum buat KALIBRASI (lihat angka saat senyum vs netral)
            smile_text = f"Senyum: {'YES :)' if is_smiling else 'NO'} | ratio={_last_smile_ratio:.2f} (th={SMILE_RATIO_THRESHOLD})"
            cv2.putText(display, smile_text,
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 220, 255), 2)

        cv2.putText(display, "Model: dlib ResNet | Smile: Landmark ratio",
                    (20, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 200, 100), 1)

        cv2.imshow("Boostify - dlib ResNet + Smile Landmark", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()