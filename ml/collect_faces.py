"""
collect_faces.py — Capture Dataset Wajah dari Kamera
======================================================
Script ini memudahkan pengumpulan foto dataset langsung
dari kamera tanpa perlu foto manual satu-satu.

Cara pakai:
  python collect_faces.py --nama "Andi Pratama" --jumlah 80

Kontrol:
  SPACE → ambil foto
  A     → auto-capture (otomatis setiap 0.5 detik)
  Q     → selesai / keluar

Output:
  dataset/raw/Andi Pratama/img_0001.jpg
  dataset/raw/Andi Pratama/img_0002.jpg
  ... dst
"""

import os
import sys
import cv2
import time
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_RAW, CAMERA_INDEX, MIN_PHOTOS_PER_PERSON
from preprocess import apply_clahe
from utils.logger import get_logger

logger = get_logger("collect_faces")


def collect_faces(nama: str, target_jumlah: int = 80):
    """
    Buka kamera, tampilkan preview, dan simpan foto
    wajah ke dataset/raw/[nama]/.

    Tips untuk hasil terbaik:
    - Ambil foto dari berbagai sudut (kiri, kanan, atas, bawah)
    - Variasikan ekspresi (datar, senyum)
    - Variasikan pencahayaan (terang, sedang, agak gelap)
    - Jarak kamera: 30-100 cm dari wajah
    """
    save_dir = os.path.join(DATASET_RAW, nama)
    os.makedirs(save_dir, exist_ok=True)

    # Hitung foto yang sudah ada
    existing = [f for f in os.listdir(save_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = len(existing)

    logger.info(f"Mengumpulkan foto untuk: {nama}")
    logger.info(f"Foto sudah ada: {count} | Target: {target_jumlah}")
    logger.info(f"Simpan ke: {save_dir}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        logger.error("Kamera tidak dapat dibuka!")
        return

    auto_capture   = False
    last_auto_time = 0
    AUTO_INTERVAL  = 0.5   # detik antar foto saat auto mode

    logger.info("\nKontrol:")
    logger.info("  SPACE → Ambil foto")
    logger.info("  A     → Toggle auto-capture")
    logger.info("  Q     → Selesai\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Tampilkan preview dengan CLAHE
        enhanced = apply_clahe(frame)

        # UI overlay
        status_color = (0, 255, 100) if auto_capture else (100, 200, 255)
        mode_text    = "AUTO" if auto_capture else "MANUAL"

        cv2.rectangle(display, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(display,
                    f"[{nama}] Foto: {count}/{target_jumlah} | Mode: {mode_text}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # Garis panduan wajah
        cx, cy = 320, 260
        cv2.ellipse(display, (cx, cy), (110, 140), 0, 0, 360, (0, 255, 100), 2)
        cv2.putText(display, "Posisikan wajah di dalam oval",
                    (130, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Progress bar
        progress = int((count / target_jumlah) * 620)
        cv2.rectangle(display, (10, 460), (630, 475), (50, 50, 50), -1)
        cv2.rectangle(display, (10, 460), (10 + progress, 475), (0, 200, 100), -1)

        cv2.imshow(f"Boostify — Kumpul Foto: {nama}", display)

        key = cv2.waitKey(1) & 0xFF

        # Cek auto capture
        now = time.time()
        should_capture = (
            key == ord(' ') or
            (auto_capture and (now - last_auto_time) >= AUTO_INTERVAL)
        )

        if should_capture and count < target_jumlah:
            fname = os.path.join(save_dir, f"img_{count+1:04d}.jpg")
            cv2.imwrite(fname, frame)
            count += 1
            last_auto_time = now
            logger.info(f"  📸 [{count}/{target_jumlah}] {fname}")

            # Flash effect
            flash = display.copy()
            cv2.rectangle(flash, (0, 0), (640, 480), (255, 255, 255), -1)
            cv2.addWeighted(flash, 0.3, display, 0.7, 0, display)
            cv2.imshow(f"Boostify — Kumpul Foto: {nama}", display)

        if key == ord('a') or key == ord('A'):
            auto_capture = not auto_capture
            logger.info(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")

        if key == ord('q') or key == ord('Q') or count >= target_jumlah:
            break

    cap.release()
    cv2.destroyAllWindows()

    logger.info(f"\n✅ Selesai! Total foto terkumpul: {count}")
    if count >= MIN_PHOTOS_PER_PERSON:
        logger.info(f"Foto sudah cukup. Sekarang jalankan:")
        logger.info(f"  1. python preprocess.py")
        logger.info(f"  2. python train.py")
    else:
        logger.warning(f"Foto masih kurang (min: {MIN_PHOTOS_PER_PERSON}). Tambah lagi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kumpulkan foto dataset wajah dari kamera"
    )
    parser.add_argument(
        "--nama", type=str, required=True,
        help='Nama orang (contoh: "Andi Pratama")'
    )
    parser.add_argument(
        "--jumlah", type=int, default=80,
        help="Target jumlah foto (default: 80)"
    )
    args = parser.parse_args()
    collect_faces(args.nama, args.jumlah)
