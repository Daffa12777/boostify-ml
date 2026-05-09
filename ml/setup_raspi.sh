#!/bin/bash
# ============================================================
# setup_raspi.sh — Setup Otomatis Boostify di Raspberry Pi
# ============================================================
# Cara pakai:
#   chmod +x setup_raspi.sh
#   ./setup_raspi.sh
# ============================================================

echo "======================================"
echo "  🐝 BOOSTIFY — Setup Raspberry Pi"
echo "======================================"

# Update sistem
echo "[1/6] Update sistem ..."
sudo apt update && sudo apt upgrade -y

# Install dependensi sistem
echo "[2/6] Install library sistem ..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqt5gui5 \
    libhdf5-dev \
    git \
    cmake

# Install library Python
echo "[3/6] Install library Python ML ..."
pip3 install --break-system-packages \
    tflite-runtime \
    numpy \
    Pillow \
    scikit-learn \
    tqdm \
    requests

# Install OpenCV headless (lebih ringan untuk Raspi)
echo "[4/6] Install OpenCV headless ..."
pip3 install --break-system-packages opencv-python-headless

# Install DeepFace (akan download model saat pertama kali dipakai)
echo "[5/6] Install DeepFace ..."
pip3 install --break-system-packages deepface==0.0.93

# Buat folder yang dibutuhkan
echo "[6/6] Buat struktur folder ..."
mkdir -p ml/dataset/raw
mkdir -p ml/dataset/processed
mkdir -p ml/dataset/test
mkdir -p ml/models
mkdir -p ml/logs

echo ""
echo "======================================"
echo "  ✅ Setup selesai!"
echo "======================================"
echo ""
echo "Langkah selanjutnya:"
echo "  1. Copy file models/*.pkl dan *.tflite dari laptop ke Raspi"
echo "  2. Jalankan: python3 iot/main.py"
echo ""

# Optional: autostart saat boot
read -p "Setup autostart saat Raspi nyala? (y/n): " yn
if [ "$yn" = "y" ]; then
    (crontab -l 2>/dev/null; echo "@reboot sleep 15 && python3 /home/pi/boostify/iot/main.py >> /home/pi/boostify/ml/logs/autostart.log 2>&1") | crontab -
    echo "✅ Autostart berhasil dikonfigurasi!"
fi

echo "Setup lengkap. Boostify siap digunakan! 🚀"
