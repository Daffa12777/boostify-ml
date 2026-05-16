"""
predict.py — Face Recognition Engine Boostify
+ Smile Detection: Simple MAR (Mouth Aspect Ratio)
  → Paling ringan, tanpa model tambahan
  → Tidak tambah beban CPU Raspberry Pi
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
from collections import deque
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    EMBEDDING_FILE, LABEL_FILE,
    SIMILARITY_THRESHOLD,
    MODEL_BACKEND,
    FACE_SIZE,
    CLAHE_CLIP, CLAHE_GRID,
    COOLDOWN_SEC,
    CAMERA_INDEX, FRAME_SKIP,
    SMILE_VOTE_FRAMES, SMILE_VOTE_THRESH
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
    "rufus" : "RFS",
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
        "time"         : now.isoformat(),
        "formattedTime": now.strftime("%A, %B %d, %Y")
    }


# ─────────────────────────────────────────────
# SMILE DETECTION — Simple MAR
# Mouth Aspect Ratio: Ringan, tanpa model tambahan
# Beban CPU: ~1ms/frame (hampir nol!)
# ─────────────────────────────────────────────

# Buffer voting multi-frame
_smile_votes = deque(maxlen=SMILE_VOTE_FRAMES)

# Threshold deteksi gigi/senyum
# Turunkan → lebih sensitif (deteksi senyum tipis)
# Naikkan  → lebih ketat (hanya senyum lebar)
SMILE_WHITE_THRESHOLD = 0.06


def _deteksi_senyum_mar(frame: np.ndarray) -> bool:
    """
    Deteksi senyum menggunakan Mouth Aspect Ratio (MAR).

    Cara kerja:
    1. Ambil area mulut dari frame (1/3 bawah tengah)
    2. Convert ke grayscale
    3. Threshold → deteksi area terang (gigi)
    4. Hitung persentase area terang
    5. Kalau > threshold → senyum (gigi terlihat)

    Kenapa efektif:
    → Saat senyum: gigi terlihat → area terang banyak
    → Saat tidak senyum: mulut tertutup → area terang sedikit

    Beban CPU: ~1ms/frame ✅
    Model tambahan: Tidak ada ✅
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Area mulut: 60-85% tinggi, 25-75% lebar frame
    y1 = int(h * 0.60)
    y2 = int(h * 0.85)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    mouth_roi = gray[y1:y2, x1:x2]

    if mouth_roi.size == 0:
        return False

    # CLAHE lokal untuk normalize pencahayaan
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    mouth_eq  = clahe.apply(mouth_roi)

    # Threshold adaptif → lebih robust terhadap cahaya
    thresh = cv2.adaptiveThreshold(
        mouth_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, -5
    )

    # Hitung persentase area terang (gigi)
    white_ratio = np.sum(thresh == 255) / thresh.size

    return white_ratio > SMILE_WHITE_THRESHOLD


def detect_smile(frame: np.ndarray) -> bool:
    """
    Deteksi senyum dengan voting multi-frame.
    Voting membuat hasil lebih stabil & tidak flickering.

    Contoh (SMILE_VOTE_FRAMES=5, SMILE_VOTE_THRESH=3):
    [True, True, False, True, True] → 4/5 → SENYUM ✅
    [False, True, False, False, True] → 2/5 → TIDAK ❌
    """
    hasil = _deteksi_senyum_mar(frame)
    _smile_votes.append(hasil)

    jumlah_senyum = sum(_smile_votes)

    if len(_smile_votes) >= SMILE_VOTE_FRAMES:
        return jumlah_senyum >= SMILE_VOTE_THRESH

    return hasil


def reset_smile_buffer():
    """Reset buffer voting setelah absen berhasil."""
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

    # ─────────────────────────────────────────
    # FUNGSI UTAMA — dipanggil IoT/predict_thread
    # ─────────────────────────────────────────
    def recognize(self, frame: np.ndarray) -> dict:
        # Step 1: enhance + crop
        enhanced = apply_clahe(frame)
        face     = get_face_crop(enhanced, self.deepface)

        # Step 2: extract embedding
        emb = self._extract_embedding(face)
        if emb is None:
            detect_smile(frame)
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
            detect_smile(frame)
            return {
                "status": "unknown", "assisstant_code": "",
                "name": "Unknown", "confidence": round(confidence, 3),
                "time": "", "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": "Wajah tidak dikenal"
            }

        # Step 5: cek cooldown
        if self._on_cooldown(nama):
            detect_smile(frame)
            return {
                "status": "cooldown", "assisstant_code": generate_code(nama),
                "name": nama, "confidence": round(confidence, 3),
                "time": "", "uuid": "", "formattedTime": "",
                "is_smiling": False, "message": f"Hai, {nama}! Sudah absen."
            }

        # Step 6: sukses
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
# TEST REALTIME
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logger.info("Test predict.py — Simple MAR Smile Detection")
    logger.info(f"Smile threshold: {SMILE_WHITE_THRESHOLD} | Vote: {SMILE_VOTE_THRESH}/{SMILE_VOTE_FRAMES}")

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

        # Visualisasi area mulut yang dianalisis
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

        # Info metode & beban
        cv2.putText(display, "Smile: Simple MAR | CPU: ~1ms",
                    (20, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 200, 100), 1)

        cv2.imshow("Boostify — Simple MAR Smile", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()