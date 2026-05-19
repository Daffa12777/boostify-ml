"""
predict_thread.py — Predict Tanpa Lag + Kirim ke Supabase
==========================================================
Disesuaikan dari predict.py yang sudah berhasil kirim ke Supabase.

Cara jalankan:
    python predict_thread.py

Tekan Q untuk keluar.
"""

import cv2
import threading
import time
import sys
import os
import json
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import FaceRecognizer
from config  import CAMERA_INDEX, FRAME_SKIP, COOLDOWN_SEC

# ─────────────────────────────────────────────
# URL BACKEND — sama persis dengan predict.py lama
# ─────────────────────────────────────────────
API_URL = "http://localhost:3000/api/uploadfromml"

# ─────────────────────────────────────────────
# SHARED STATE antar thread
# ─────────────────────────────────────────────
latest_frame    = None
latest_result   = {}
lock            = threading.Lock()
running         = True

# Cooldown upload per orang (sama dengan predict.py lama)
last_upload_time = {}


# ─────────────────────────────────────────────
# KIRIM KE SUPABASE — sama persis dengan predict.py lama
# ─────────────────────────────────────────────
def kirim_ke_backend(result: dict, nama: str):
    """
    Kirim data ke backend → masuk Supabase.
    Logika sama persis dengan predict.py lama yang sudah berhasil.
    """
    global last_upload_time

    current_time = time.time()

    # Inisialisasi kalau belum ada
    if nama not in last_upload_time:
        last_upload_time[nama] = 0

    # Cek cooldown manual
    if current_time - last_upload_time[nama] > COOLDOWN_SEC:

        print("\n" + "=" * 50)
        print("DATA ABSENSI (siap kirim ke API):")
        print(json.dumps({
            "assisstant_code": result["assisstant_code"],
            "name"           : result["name"],
            "time"           : result["time"],
            "uuid"           : result["uuid"],
            "formattedTime"  : result["formattedTime"],
            "is_smiling"     : result["is_smiling"]
        }, indent=4))
        print("=" * 50)

        try:
            # Kirim full result — sama dengan predict.py lama
            response = requests.post(
                API_URL,
                json=result
            )
            print("API RESPONSE:", response.text)

            # Update waktu upload terakhir
            last_upload_time[nama] = current_time

        except Exception as e:
            print("GAGAL KIRIM KE BACKEND:", e)


# ─────────────────────────────────────────────
# THREAD 1 — Kamera (terus capture tanpa henti)
# ─────────────────────────────────────────────
def thread_kamera():
    global latest_frame, running

    print("[kamera] Membuka kamera ...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # buffer kecil = tidak lag
    time.sleep(1)

    if not cap.isOpened():
        print("[kamera] ❌ Kamera tidak bisa dibuka!")
        running = False
        return

    print("[kamera] ✅ Kamera siap.")

    while running:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame.copy()
        else:
            time.sleep(0.01)

    cap.release()
    print("[kamera] Kamera ditutup.")


# ─────────────────────────────────────────────
# THREAD 2 — ML + Kirim ke Supabase
# ─────────────────────────────────────────────
def thread_ml(recognizer):
    global latest_result, running

    frame_count = 0
    print("[ml] Thread ML siap.")

    while running:
        # Ambil frame terbaru
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # Skip frame → hemat CPU
        if frame_count % FRAME_SKIP != 0:
            time.sleep(0.005)
            continue

        # Proses ML
        result = recognizer.recognize(frame)

        # Simpan hasil untuk ditampilkan
        with lock:
            latest_result = result

        # Kirim ke Supabase kalau berhasil absen
        if result["status"] == "recognized":
            nama = result.get("name", "")
            kirim_ke_backend(result, nama)


# ─────────────────────────────────────────────
# MAIN — Tampilan kamera (main thread)
# ─────────────────────────────────────────────
def main():
    global running

    print("=" * 50)
    print("  🐝 BOOSTIFY — No Lag + Kirim ke Supabase")
    print("=" * 50)

    # Load model
    print("\n[main] Loading model ML ...")
    try:
        recognizer = FaceRecognizer()
    except FileNotFoundError:
        print("[main] ❌ Model tidak ditemukan! Jalankan train.py dulu.")
        return

    # Jalankan 2 thread
    t_kamera = threading.Thread(target=thread_kamera, daemon=True)
    t_ml     = threading.Thread(target=thread_ml, args=(recognizer,), daemon=True)

    t_kamera.start()
    t_ml.start()

    # Tunggu kamera siap
    time.sleep(2)
    print("\n[main] ✅ Sistem siap! Tekan Q untuk keluar.\n")

    # ── Loop tampilan — sama dengan predict.py lama ──
    while True:
        with lock:
            frame  = latest_frame.copy()  if latest_frame  is not None else None
            result = latest_result.copy() if latest_result is not None else {}

        if frame is None:
            time.sleep(0.01)
            continue

        # ── Tampilan kamera ──
        display = frame.copy()

        if result:
            status     = result.get("status", "")
            nama       = result.get("name", "")
            conf       = result.get("confidence", 0.0)
            code       = result.get("assisstant_code", "")
            is_smiling = result.get("is_smiling", False)

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
                smile_text = "Senyum: YES :)" if is_smiling else "Senyum: NO"
                cv2.putText(display, smile_text,
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 220, 255), 2)

                cv2.putText(display, result.get("formattedTime", ""),
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)

        # Info threading
        cv2.putText(display, "Threading: ON | No Lag",
                    (20, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 100, 100), 1)

        cv2.imshow("Boostify — Recognition + Smile", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()
    print("\n[main] Sistem dihentikan.")


if __name__ == "__main__":
    main()