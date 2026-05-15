# 🐝 Boostify ML — Face Recognition Attendance System

Sistem absensi otomatis berbasis **pengenalan wajah** dan **deteksi senyum** yang berjalan di atas **Raspberry Pi 5**.

---

## 📋 Deskripsi

Boostify ML adalah modul machine learning dari project Boostify — sistem absensi mahasiswa berbasis kamera. Sistem ini mampu:

- ✅ Mengenali wajah mahasiswa secara realtime
- ✅ Mendeteksi senyum/tidak senyum
- ✅ Mendaftarkan mahasiswa baru tanpa training ulang dari awal
- ✅ Berjalan ringan di Raspberry Pi 5
- ✅ Output langsung match dengan API backend `/api/attendances`

---

## 🧠 Teknologi

| Komponen | Teknologi |
|---|---|
| Face Recognition | GhostFaceNet (via DeepFace) |
| Face Detection | OpenCV (ringan untuk Raspi) |
| Smile Detection | OpenCV Haar Cascade |
| Preprocessing | CLAHE Enhancement |
| Augmentasi | Flip, Brightness, Rotasi, Noise (15x) |
| Similarity | Cosine Similarity |
| Version Control | Git + GitHub |

---

## 📁 Struktur File

```
ml/
├── dataset/
│   ├── raw/[nama]/         ← foto mentah per orang
│   ├── processed/[nama]/   ← hasil preprocessing
│   └── test/[nama]/        ← foto untuk evaluasi
│
├── models/
│   ├── embeddings.pkl      ← database embedding wajah
│   └── labels.pkl          ← daftar nama terdaftar
│
├── utils/
│   ├── __init__.py
│   └── logger.py           ← sistem logging UTF-8
│
├── config.py               ← ⚙️ semua konfigurasi di sini
├── collect_faces.py        ← 📸 ambil foto dataset dari kamera
├── preprocess.py           ← 🔧 CLAHE + crop + augmentasi
├── train.py                ← 🎓 training & generate embeddings
├── predict.py              ← 🎯 engine recognition (dipanggil IoT)
├── evaluate.py             ← 📊 uji akurasi model
├── requirements.txt        ← 📦 daftar library
└── setup_raspi.sh          ← 🔨 setup otomatis di Raspi
```

---

## ⚙️ Konfigurasi (`config.py`)

| Parameter | Value | Keterangan |
|---|---|---|
| `MODEL_BACKEND` | `GhostFaceNet` | Model ringan + akurat untuk Raspi |
| `DETECTOR_BACKEND` | `opencv` | Detector ringan |
| `SIMILARITY_THRESHOLD` | `0.55` | Threshold kecocokan wajah |
| `MIN_PHOTOS_PER_PERSON` | `50` | Minimal foto per orang |
| `AUGMENT_PER_IMAGE` | `15` | 50 foto × 15 = 750 data |
| `FRAME_SKIP` | `5` | Proses 1 dari 5 frame (hemat CPU) |
| `COOLDOWN_SEC` | `20` | Jeda setelah absen berhasil |

---

## 🚀 Cara Pakai

### Install Dependencies
```bash
pip install -r requirements.txt
```

### 1. Kumpulkan Dataset
```bash
python collect_faces.py --nama "Nama Mahasiswa" --jumlah 50
```
Tekan **A** untuk auto-capture, gerakkan kepala pelan-pelan.

### 2. Preprocessing
```bash
python preprocess.py
```

### 3. Training
```bash
# Training pertama kali (semua orang)
python train.py

# Daftarkan mahasiswa BARU tanpa train ulang semua
python train.py --register "Nama Mahasiswa Baru"
```

### 4. Evaluasi
```bash
python evaluate.py
```

### 5. Test Realtime
```bash
python predict.py
```

---

## 📡 Output Format (untuk Tim IoT & Web)

Setiap absen berhasil, `predict.py` mengembalikan:

```python
{
    "status"          : "recognized",
    "assisstant_code" : "FDR",        # kode unik max 3 huruf
    "name"            : "daffa",
    "confidence"      : 0.877,
    "time"            : "2026-05-09T14:18:54.737Z",
    "uuid"            : "5461f949-66a8-4409-bc2f-ef42b6126a5d",
    "formattedTime"   : "Saturday, May 09, 2026",
    "is_smiling"      : True,
    "message"         : "Selamat Datang, daffa! Senyumnya Keren!"
}
```

Format ini **langsung match** dengan API `/api/attendances`.

---

## 👥 Cara Integrasi dengan Tim IoT

```python
from ml.predict import FaceRecognizer

# Load sekali di awal
recognizer = FaceRecognizer()

# Panggil tiap frame
result = recognizer.recognize(frame)

if result["status"] == "recognized":
    print(result["name"])      # nama mahasiswa
    print(result["message"])   # pesan untuk LCD
    print(result["is_smiling"])# status senyum
```

---

## ➕ Daftarkan Mahasiswa Baru

Tidak perlu training ulang semua data!

```bash
# 1. Ambil foto
python collect_faces.py --nama "Mahasiswa Baru" --jumlah 50

# 2. Preprocessing
python preprocess.py

# 3. Register saja (± 30 detik)
python train.py --register "Mahasiswa Baru"
```

---

## 🔧 Kode Asisten Custom

Edit `KODE_ASISTEN` di `predict.py`:

```python
KODE_ASISTEN = {
    "daffa" : "FDR",
    "alif"  : "ALF",
    "rufus" : "RFS",
    # tambah mahasiswa baru:
    # "nama" : "KOD",  ← max 3 huruf!
}
```

---

## 📊 Hasil Evaluasi

| Metrik | Nilai |
|---|---|
| Model | GhostFaceNet |
| Akurasi | 90-94% |
| Confidence Score | 0.8 – 0.9 |
| Threshold | 0.55 |
| Dataset | 50 foto × 15 aug = 750 data/orang |
| Waktu daftar | ~2 menit/mahasiswa |
| Status | ✅ Siap Deploy ke Raspberry Pi |

---

## 🔄 Perbandingan Model

| Aspek | Facenet (Lama) | GhostFaceNet (Baru) |
|---|---|---|
| Akurasi (50 foto) | 85-90% | 90-94% ✅ |
| Kecepatan Raspi | ~80ms | ~60ms ✅ |
| Ukuran model | ~90MB | ~20MB ✅ |
| Foto per orang | 200 foto | 50 foto ✅ |
| Waktu/mahasiswa | ~6 menit | ~2 menit ✅ |
| 100 mahasiswa | ~10 jam | ~3.5 jam ✅ |

---

## 💡 Solusi Masalah Umum

| Masalah | Solusi |
|---|---|
| Kamera tidak terbuka | Tambahkan `cv2.CAP_DSHOW` (Windows) |
| Error `/tmp/` di Windows | Sudah difix pakai `tempfile.gettempdir()` |
| Wajah tidak terdeteksi | Pastikan pencahayaan cukup, CLAHE aktif |
| Raspi kepanasan | Naikkan `FRAME_SKIP` di config.py |
| Shapes not aligned | Hapus `.pkl` lama, register ulang |

---

## 🔗 Deploy ke Raspberry Pi

```bash
# Di Raspberry Pi
git clone https://github.com/Daffa12777/boostify-ml.git
cd boostify-ml
chmod +x ml/setup_raspi.sh
./ml/setup_raspi.sh
```

---

## 📦 Update ke GitHub

```bash
# Setiap ada perubahan kode
git add .
git commit -m "keterangan perubahan"
git push

# Setelah daftar mahasiswa baru
git add ml/models/
git commit -m "tambah mahasiswa baru - retrain"
git push
```

---

## 👤 Tim

**Machine Learning** — Muhammad Daffa  
**Repository** — [github.com/Daffa12777/boostify-ml](https://github.com/Daffa12777/boostify-ml)  
**Project** — Boostify Face Recognition Attendance System

---

*Boostify ML — Smart Attendance for Smart Campus* 🐝