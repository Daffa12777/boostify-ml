
"""
main.py — BOOSTIFY FINAL VERSION
Raspberry Pi 5 + ML + LCD + Speaker + Web Integration
"""

import sys
import os
import time
import signal
import requests
import json
import random
from datetime import datetime

# =========================================================
# PATH ML
# =========================================================
ML_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ML_PATH)

from predict import FaceRecognizer
from config import (
    COOLDOWN_SEC,
    CAMERA_INDEX,
    FRAME_SKIP
)

import cv2
import pygame

# =========================================================
# CONFIG
# =========================================================
API_URL = "http://localhost:3000/api/uploadfromml"
IDLE_TIMEOUT = 10

# =========================================================
# INIT PYGAME LCD
# =========================================================
pygame.init()

LCD_W = 480
LCD_H = 320

screen = pygame.display.set_mode((LCD_W, LCD_H))
pygame.display.set_caption("BOOSTIFY")

font_nama  = pygame.font.SysFont("Arial", 38, bold=True)
font_pesan = pygame.font.SysFont("Arial", 26)
font_kecil = pygame.font.SysFont("Arial", 18)

# =========================================================
# PESAN MOTIVASI
# =========================================================
PESAN = {
    "Pagi": [
        "Semangat kuliahnya! ☀️",
        "Hari ini pasti produktif!",
        "Jangan lupa sarapan!"
    ],

    "Siang": [
        "Jangan lupa makan siang!",
        "Tetap fokus ya 😄",
        "Semangat terus!"
    ],

    "Sore": [
        "Good job hari ini 💪",
        "Hampir selesai!",
        "Tetap semangat!"
    ],

    "Malam": [
        "Jangan begadang 🌙",
        "Istirahat yang cukup!",
        "Kerja keras terbayar!"
    ]
}

# =========================================================
# AUDIO
# =========================================================
try:
    import subprocess
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False

def putar_suara(teks):
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

# =========================================================
# WAKTU
# =========================================================
def get_waktu():
    hour = datetime.now().hour

    if 5 <= hour < 12:
        return "Pagi"

    elif 12 <= hour < 15:
        return "Siang"

    elif 15 <= hour < 18:
        return "Sore"

    else:
        return "Malam"

# =========================================================
# LCD UI
# =========================================================
def tampilkan_standby():

    screen.fill((10, 10, 20))

    pygame.draw.circle(screen, (40, 40, 90), (240, 90), 60)

    title = font_nama.render("BOOSTIFY", True, (255, 220, 50))
    screen.blit(title, title.get_rect(center=(240, 90)))

    text = font_pesan.render(
        "Arahkan wajah ke kamera...",
        True,
        (180, 180, 255)
    )

    screen.blit(text, text.get_rect(center=(240, 200)))

    pygame.display.flip()

def tampilkan_greeting(name, confidence, is_smiling):

    waktu = get_waktu()

    motivasi = random.choice(PESAN[waktu])

    nama_display = name.capitalize()

    screen.fill((20, 20, 40))

    pygame.draw.circle(screen, (50, 50, 100), (240, 80), 60)

    sapaan = f"Selamat {waktu},"

    t1 = font_pesan.render(
        sapaan,
        True,
        (180, 180, 255)
    )

    screen.blit(t1, t1.get_rect(center=(240, 150)))

    t2 = font_nama.render(
        nama_display + "!",
        True,
        (255, 220, 50)
    )

    screen.blit(t2, t2.get_rect(center=(240, 195)))

    pesan = (
        "Senyumnya Keren 😄"
        if is_smiling
        else motivasi
    )

    t3 = font_pesan.render(
        pesan,
        True,
        (150, 255, 150)
    )

    screen.blit(t3, t3.get_rect(center=(240, 245)))

    conf_text = f"Confidence: {confidence*100:.1f}%"

    t4 = font_kecil.render(
        conf_text,
        True,
        (120, 120, 120)
    )

    screen.blit(t4, t4.get_rect(center=(240, 295)))

    pygame.display.flip()

def tampilkan_unknown():

    screen.fill((70, 0, 0))

    text1 = font_nama.render(
        "Tidak Dikenal",
        True,
        (255, 255, 255)
    )

    screen.blit(text1, text1.get_rect(center=(240, 140)))

    text2 = font_pesan.render(
        "Coba Lagi",
        True,
        (255, 180, 180)
    )

    screen.blit(text2, text2.get_rect(center=(240, 210)))

    pygame.display.flip()

# =========================================================
# API ATTENDANCE
# =========================================================
last_upload_time = {}

def kirim_absen(result):

    nama = result.get("name", "")

    current_time = time.time()

    if nama not in last_upload_time:
        last_upload_time[nama] = 0

    if current_time - last_upload_time[nama] <= COOLDOWN_SEC:
        return

    try:

        response = requests.post(
            API_URL,
            json=result,
            timeout=5
        )

        print(f"[API] {response.status_code}")

        last_upload_time[nama] = current_time

    except Exception as e:
        print(f"[API ERROR] {e}")

# =========================================================
# MAIN
# =========================================================
def main():

    print("=" * 50)
    print("BOOSTIFY FINAL SYSTEM")
    print("=" * 50)

    recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    time.sleep(1)

    if not cap.isOpened():
        print("Kamera gagal dibuka")
        return

    tampilkan_standby()

    putar_suara("Boostify siap digunakan")

    frame_count = 0

    last_action_time = time.time()

    def handle_exit(sig, frame):

        cap.release()

        pygame.quit()

        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                handle_exit(None, None)

        ret, frame = cap.read()

        if not ret:
            continue

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        result = recognizer.recognize(frame)

        status = result.get("status", "no_face")

        # =================================================
        # RECOGNIZED
        # =================================================
        if status == "recognized":

            nama       = result["name"]
            confidence = result["confidence"]

            is_smiling = result.get("is_smiling", False)

            tampilkan_greeting(
                nama,
                confidence,
                is_smiling
            )

            pesan_audio = (
                f"Selamat datang {nama}, senyumnya keren"
                if is_smiling
                else f"Selamat datang {nama}"
            )

            putar_suara(pesan_audio)

            kirim_absen(result)

            last_action_time = time.time()

            print(f"[ABSEN] {nama}")

        # =================================================
        # UNKNOWN
        # =================================================
        elif status == "unknown":

            tampilkan_unknown()

            putar_suara("Wajah tidak dikenal")

            print("[UNKNOWN FACE]")

        # =================================================
        # IDLE
        # =================================================
        elif status == "no_face":

            if time.time() - last_action_time > IDLE_TIMEOUT:
                tampilkan_standby()

if __name__ == "__main__":
    main()

