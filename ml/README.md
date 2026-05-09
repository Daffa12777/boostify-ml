# 🧠 Boostify — Tim ML Documentation

Sistem absensi wajah berbasis **Face Recognition** menggunakan **DeepFace + ArcFace**.

---

## 📁 Struktur File

```
ml/
├── dataset/
│   ├── raw/              ← foto mentah per orang (INPUT)
│   │   ├── Andi/
│   │   │   ├── img_0001.jpg
│   │   │   └── ...
│   │   └── Budi/
│   ├── processed/        ← hasil preprocessing (AUTO-GENERATED)
│   └── test/             ← foto untuk evaluasi (pisah dari training)
│       ├── Andi/
│       └── Budi/
│
├── models/               ← output training (dikirim ke Raspi)
│   ├── embeddings.pkl    ← database embedding semua wajah
│   └── labels.pkl        ← daftar nama terdaftar
│
├── utils/
│   ├── __init__.py
│   └── logger.py         ← sistem logging
│
├── config.py             ← ⚙️ SEMUA konfigurasi di sini
├── collect_faces.py      ← 📸 Kumpul foto dataset dari kamera
├── preprocess.py         ← 🔧 CLAHE + crop + augmentasi
├── train.py              ← 🎓 Training & generate embeddings
├── predict.py            ← 🎯 Inferensi (dipanggil IoT)
├── evaluate.py           ← 📊 Uji akurasi model
├── requirements.txt      ← 📦 Daftar library
└── setup_raspi.sh        ← 🔨 Setup otomatis di Raspi
```

---

## 🚀 Alur Kerja Tim ML

```
STEP 1: Kumpul Data
  → python collect_faces.py --nama "Nama Orang" --jumlah 80

STEP 2: Preprocessing  
  → python preprocess.py

STEP 3: Training
  → python train.py

STEP 4: Evaluasi (wajib sebelum deploy!)
  → python evaluate.py
  Target: akurasi ≥ 90%

STEP 5: Deploy ke Raspi
  → Copy folder models/ ke Raspberry Pi
  → Tim IoT yang menjalankan main.py
```

---

## ⚙️ Konfigurasi Penting (`config.py`)

| Parameter | Default | Keterangan |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.60` | Threshold kecocokan wajah |
| `FRAME_SKIP` | `5` | Proses 1 dari setiap N frame |
| `COOLDOWN_SEC` | `3` | Jeda detik setelah absen |
| `AUGMENT_PER_IMAGE` | `5` | Jumlah augmentasi per foto |
| `MODEL_BACKEND` | `ArcFace` | Model recognition |
| `DETECTOR_BACKEND` | `mtcnn` | Model deteksi wajah |

---

## 📡 Interface dengan Tim IoT

Tim IoT **hanya perlu import 1 class** dari Tim ML:

```python
# Di main.py (Tim IoT)
from ml.predict import FaceRecognizer

# Inisialisasi SEKALI di awal
recognizer = FaceRecognizer()

# Panggil tiap frame
result = recognizer.recognize(frame)

# Format return:
# {
#   "status"    : "recognized" | "unknown" | "no_face" | "cooldown",
#   "nama"      : "Andi Pratama",
#   "confidence": 0.92,
#   "message"   : "Selamat Datang, Andi! Semangat! 💪"
# }
```

---

## 💡 Tips Dataset yang Baik

- **Minimal 50 foto per orang** (80+ lebih baik)
- Variasikan **sudut**: lurus, kiri 30°, kanan 30°, atas, bawah
- Variasikan **pencahayaan**: terang, sedang, agak gelap
- Variasikan **ekspresi**: netral, senyum, serius
- Jarak kamera: **30–80 cm** dari wajah
- Pisahkan **10–20 foto** untuk folder `dataset/test/` (jangan dipakai training)

---

## 🔧 Troubleshooting

**Model tidak mengenali wajah:**
- Turunkan `SIMILARITY_THRESHOLD` di config.py (misal dari 0.60 ke 0.50)
- Tambah lebih banyak foto training
- Pastikan pencahayaan saat training mirip dengan kondisi real

**Raspi terlalu panas:**
- Naikkan `FRAME_SKIP` (misal dari 5 ke 10)
- Pastikan kipas terpasang dan heatsink ada
- Buat lubang ventilasi di casing akrilik

**Wajah tidak terdeteksi di kondisi gelap:**
- CLAHE sudah diaplikasikan secara otomatis
- Tambah lampu tambahan di sekitar kamera
- Pertimbangkan kamera NoIR + IR LED
