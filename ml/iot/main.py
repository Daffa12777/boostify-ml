"""
main.py — File Utama Boostify di Raspberry Pi 5
================================================
Hardware (sesuai skematik):
  - Raspberry Pi 5
  - USB WebCam
  - LCD 3.5" TFT 480x320 (ILI9486) via SPI
  - USB Speaker
  - Cooling Fan

Cara jalankan:
  python3 main.py

Auto-start saat boot:
  crontab -e
  @reboot sleep 15 && python3 /home/pi/boostify-ml/ml/iot/main.py
"""

import sys
import os
import time
import signal
import requests
import json

# ── Path ke folder ml/ (parent dari iot/)
ML_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ML_PATH)

from predict import FaceRecognizer
from config  import COOLDOWN_SEC, CAMERA_INDEX, FRAME_SKIP

import cv2

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

# URL Backend — ganti dengan URL dari Tim Web Dev
API_URL      = "http://localhost:3000/api/uploadfromml"
IDLE_TIMEOUT = 10   # detik sebelum LCD kembali idle

print("=" * 50)
print("  🐝  BOOSTIFY — Sistem Absensi Wajah")
print("  Raspberry Pi 5 + LCD 480x320")
print("=" * 50)


# ─────────────────────────────────────────────
# IMPORT KOMPONEN IoT
# Pakai try/except supaya tidak error di laptop
# ─────────────────────────────────────────────

# ── LCD ──
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[lcd] Pillow tidak ada. pip install Pillow")

try:
    from luma.lcd.device import ili9486
    from luma.core.interface.serial import spi
    serial = spi(port=0, device=0, gpio_DC=25, gpio_RST=24)
    lcd = ili9486(serial, rotate=1)
    LCD_OK = True
    print("[lcd] ✅ LCD ILI9486 siap.")
except Exception:
    LCD_OK = False
    print("[lcd] LCD tidak tersedia (normal di laptop).")

# ── Audio ──
try:
    import subprocess
    AUDIO_OK = True
    print("[audio] ✅ USB Speaker siap.")
except Exception:
    AUDIO_OK = False


# ─────────────────────────────────────────────
# FUNGSI LCD
# ─────────────────────────────────────────────
LCD_W, LCD_H = 480, 320

def _get_font(size):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _tampil_lcd(baris1, baris2, baris3="", bg=(0,0,0)):
    if not PIL_OK:
        return
    img  = Image.new('RGB', (LCD_W, LCD_H), color=bg)
    draw = ImageDraw.Draw(img)
    cx   = LCD_W // 2

    # Garis dekorasi
    draw.rectangle([(0,0),(LCD_W,6)],   fill=(245,197,24))
    draw.rectangle([(0,LCD_H-6),(LCD_W,LCD_H)], fill=(245,197,24))

    if baris1:
        draw.text((cx, 80),  baris1, font=_get_font(24), fill=(255,220,0),   anchor="mm")
    if baris2:
        draw.text((cx, 160), baris2, font=_get_font(36), fill=(255,255,255), anchor="mm")
    if baris3:
        draw.text((cx, 240), baris3, font=_get_font(22), fill=(0,220,255),   anchor="mm")

    if LCD_OK:
        lcd.display(img)

def tampil_idle():
    print("[lcd] 💤 Idle")
    _tampil_lcd("🐝 BOOSTIFY", "Silakan Absen", "Hadapkan wajah ke kamera", (10,10,30))

def tampil_berhasil(nama, pesan, is_smiling):
    print(f"[lcd] ✅ {nama} — {pesan}")
    _tampil_lcd("Selamat Datang!", nama, pesan, (0,120,50))

def tampil_gagal():
    print("[lcd] ❌ Tidak dikenal")
    _tampil_lcd("Tidak Dikenal", "Coba Lagi", "", (180,0,0))

def tampil_loading(pesan="Memuat sistem..."):
    print(f"[lcd] ⏳ {pesan}")
    _tampil_lcd("Boostify", pesan, "Mohon tunggu...", (0,50,100))


# ─────────────────────────────────────────────
# FUNGSI AUDIO
# ─────────────────────────────────────────────
def putar_suara(teks):
    """Text-to-speech pakai espeak."""
    if not AUDIO_OK:
        return
    try:
        subprocess.Popen(
            ["espeak", "-v", "id", "-s", "140", teks],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


# ─────────────────────────────────────────────
# FUNGSI KIRIM KE BACKEND
# ─────────────────────────────────────────────
last_upload_time = {}

def kirim_absen(result: dict):
    """Kirim data absensi ke backend → Supabase."""
    nama         = result.get("name", "")
    current_time = time.time()

    if nama not in last_upload_time:
        last_upload_time[nama] = 0

    # Cek cooldown anti spam
    if current_time - last_upload_time[nama] <= COOLDOWN_SEC:
        return

    print("\n" + "=" * 50)
    print("DATA ABSENSI:")
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
        response = requests.post(
            API_URL,
            json    = result,
            timeout = 5
        )
        print(f"[api] ✅ Terkirim! Response: {response.status_code}")
        last_upload_time[nama] = current_time
    except Exception as e:
        print(f"[api] ❌ Gagal kirim: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Startup
    tampil_loading("Memuat model ML...")

    # Load model
    print("[main] Loading model ML ...")
    try:
        recognizer = FaceRecognizer()
    except FileNotFoundError:
        print("[main] ❌ Model tidak ditemukan! Jalankan train.py dulu.")
        tampil_loading("ERROR: Model tidak ada!")
        return

    # Buka kamera
    print("[main] Membuka kamera ...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    time.sleep(1)

    if not cap.isOpened():
        print("[main] ❌ Kamera tidak bisa dibuka!")
        tampil_loading("ERROR: Kamera gagal!")
        return

    tampil_idle()
    putar_suara("Boostify siap, silakan absen")
    print("[main] ✅ Sistem siap!\n")

    frame_count      = 0
    last_status      = ""
    last_action_time = time.time()

    # Handle Ctrl+C
    def handle_exit(sig, frame):
        print("\n[main] Menghentikan sistem ...")
        cap.release()
        print("[main] Sampai jumpa! 🐝")
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # ── Loop Utama ──
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        # Proses ML
        result = recognizer.recognize(frame)
        status = result.get("status", "no_face")

        # ── Handle hasil ──
        if status == "recognized":
            nama       = result["name"]
            is_smiling = result.get("is_smiling", False)
            pesan      = "Senyumnya Keren!" if is_smiling else "Semangat!"

            print(f"[main] ✅ ABSEN: {nama} | conf: {result['confidence']} | senyum: {is_smiling}")

            # Tampil LCD
            tampil_berhasil(nama, pesan, is_smiling)

            # Suara
            putar_suara(f"Selamat datang {nama}, {pesan}")

            # Kirim ke Supabase
            kirim_absen(result)

            last_action_time = time.time()
            last_status      = "recognized"

        elif status == "unknown":
            print(f"[main] ❌ Tidak dikenal | conf: {result['confidence']}")
            if last_status != "unknown":
                tampil_gagal()
                putar_suara("Wajah tidak dikenal, coba lagi")
            last_status = "unknown"

        elif status == "cooldown":
            if last_status != "cooldown":
                print(f"[main] ⏳ {result['name']} sudah absen.")
            last_status = "cooldown"

        elif status == "no_face":
            if time.time() - last_action_time > IDLE_TIMEOUT:
                if last_status != "idle":
                    tampil_idle()
                    last_status = "idle"


if __name__ == "__main__":
    main()