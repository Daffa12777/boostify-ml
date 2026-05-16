# 🐝 Boostify ML — Face Recognition Attendance System

Sistem absensi otomatis berbasis **pengenalan wajah** dan **deteksi senyum** yang berjalan di atas **Raspberry Pi 5**.

---

## 📋 Deskripsi

Boostify ML adalah modul machine learning dari project Boostify — sistem absensi mahasiswa berbasis kamera. Sistem ini mampu:

- ✅ Mengenali wajah mahasiswa secara realtime
- ✅ Mendeteksi senyum/tidak senyum (Simple MAR — ringan tanpa model tambahan)
- ✅ Mendaftarkan mahasiswa baru tanpa training ulang dari awal
- ✅ Berjalan ringan di Raspberry Pi 5 (tidak overheat)
- ✅ Kamera tidak lag (threading)
- ✅ Output langsung match dengan API backend `/api/uploadfromml`
- ✅ Anti spam cooldown (1 orang 1x per 60 detik)

---

## 🧠 Model yang Digunakan (Total: 2 Model)

| No | Model | Fungsi | Ukuran | Beban CPU |
|---|---|---|---|---|
| 1 | **GhostFaceNet** (via DeepFace) | Face Recognition — kenali siapa orangnya | ~20MB | ~60ms/frame |
| 2 | **OpenCV Haar Cascade** | Face Detection — deteksi ada wajah atau tidak | Built-in | ~5ms/frame |
| - | **Simple MAR** | Smile Detection — bukan model, murni image processing | 0MB | ~1ms/frame |

> **Catatan:** Simple MAR (Mouth Aspect Ratio) bukan model ML — hanya algoritma pengolahan gambar berbasis persentase kecerahan area mulut. Sangat ringan dan tidak menambah beban Raspberry Pi.

---

## 🔄 Perubahan dari Versi Sebelumnya

| Komponen | Sebelumnya | Sekarang | Alasan |
|---|---|---|---|
| Face Recognition | Facenet | **GhostFaceNet** | Lebih akurat untuk sedikit foto, lebih ringan |
| Face Detector | mtcnn | **OpenCV** | Lebih ringan di Raspi |
| Smile Detection | Haar Cascade | **Simple MAR** | Haar Cascade tidak reliable, MAR lebih akurat & ringan |
| Dataset | 200 foto | **50 foto** | Cukup dengan augmentasi 15x |
| Kamera | Lag (sequential) | **No Lag (threading)** | Thread terpisah untuk kamera & ML |
| JSON bool | numpy bool | **Python bool** | Fix TypeError JSON serializable |
| config.py | Ada typo `"INFO".` | **`"INFO"`** | Fix syntax error |
| preprocess.py | Ada dead code | **Dead code dihapus** | Fix logika deteksi wajah |

---

## 📁 Struktur File

```
ml/
├── dataset/
│   ├── raw/[nama]/           ← foto mentah per orang
│   ├── processed/[nama]/     ← hasil preprocessing
│   └── test/[nama]/          ← foto untuk evaluasi
│
├── models/
│   ├── embeddings.pkl        ← database embedding wajah (GhostFaceNet)
│   └── labels.pkl            ← daftar nama terdaftar
│
├── utils/
│   ├── __init__.py
│   └── logger.py             ← sistem logging UTF-8
│
├── config.py                 ← ⚙️ semua konfigurasi di sini
├── collect_faces.py          ← 📸 ambil foto dataset dari kamera
├── preprocess.py             ← 🔧 CLAHE + crop + augmentasi (fixed)
├── train.py                  ← 🎓 training & generate embeddings
├── predict.py                ← 🎯 engine recognition + smile detection
├── predict_thread.py         ← 🚀 runner no-lag (pakai ini!)
├── evaluate.py               ← 📊 uji akurasi model
├── requirements.txt          ← 📦 daftar library
└── setup_raspi.sh            ← 🔨 setup otomatis di Raspi
```

---

## ⚙️ Konfigurasi (`config.py`)

| Parameter | Value | Keterangan |
|---|---|---|
| `MODEL_BACKEND` | `GhostFaceNet` | Model face recognition |
| `DETECTOR_BACKEND` | `opencv` | Detector wajah (ringan) |
| `SIMILARITY_THRESHOLD` | `0.55` | Threshold kecocokan wajah |
| `MIN_PHOTOS_PER_PERSON` | `50` | Minimal foto per orang |
| `AUGMENT_PER_IMAGE` | `15` | 50 foto × 15 = 750 data |
| `FRAME_SKIP` | `8` | Proses 1 dari 8 frame (hemat CPU) |
| `COOLDOWN_SEC` | `60` | Jeda absen per orang (anti spam) |
| `SMILE_VOTE_FRAMES` | `5` | Frame voting untuk smile detection |
| `SMILE_VOTE_THRESH` | `3` | Minimal frame senyum dari voting |
| `SMILE_MIN_NEIGHBORS` | `8` | Sensitivitas deteksi |

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

### 5. Jalankan Sistem (Gunakan ini!)
```bash
# Versi NO LAG (direkomendasikan)
python predict_thread.py

# Versi biasa (bisa lag)
python predict.py
```

---

## 🎯 Cara Kerja Smile Detection (Simple MAR)

```
Frame kamera masuk
        ↓
Ambil area mulut (60-85% tinggi, 25-75% lebar frame)
        ↓
CLAHE lokal → normalize pencahayaan
        ↓
Adaptive Threshold → deteksi area terang (gigi)
        ↓
Hitung persentase area terang
        ↓
> 6% area terang → SENYUM ✅
< 6% area terang → TIDAK SENYUM ❌
        ↓
Voting 5 frame → hasil lebih stabil
```

> Untuk tuning: ubah `SMILE_WHITE_THRESHOLD = 0.06` di `predict.py`
> - Turunkan (0.04) → lebih sensitif
> - Naikkan (0.08) → lebih ketat

---

## 📡 Output Format (untuk Tim IoT & Web)

Setiap absen berhasil, `predict.py` mengembalikan:

```json
{
    "status"          : "recognized",
    "assisstant_code" : "FDR",
    "name"            : "daffa",
    "confidence"      : 0.877,
    "time"            : "2026-05-09T14:18:54.737Z",
    "uuid"            : "5461f949-66a8-4409-bc2f-ef42b6126a5d",
    "formattedTime"   : "Saturday, May 09, 2026",
    "is_smiling"      : true,
    "message"         : "Selamat Datang, daffa! Senyumnya Keren!"
}
```

Dikirim ke backend via:
```python
requests.post("http://localhost:3000/api/uploadfromml", json=result)
```

---

## 👥 Cara Integrasi dengan Tim IoT

```python
from ml.predict import FaceRecognizer

# Load sekali di awal
recognizer = FaceRecognizer()

# Panggil tiap frame
result = recognizer.recognize(frame)

if result["status"] == "recognized":
    print(result["name"])       # nama mahasiswa
    print(result["message"])    # pesan untuk LCD
    print(result["is_smiling"]) # status senyum (bool)
```

---

## ➕ Daftarkan Mahasiswa Baru (Tanpa Train Ulang)

```bash
# 1. Ambil foto (hanya ~45 detik)
python collect_faces.py --nama "Mahasiswa Baru" --jumlah 50

# 2. Preprocessing (~30 detik)
python preprocess.py

# 3. Register saja (~30 detik)
python train.py --register "Mahasiswa Baru"

# Total: ~2 menit per mahasiswa ✅
```

---

## 🔧 Kode Asisten Custom

Edit `KODE_ASISTEN` di `predict.py`:

```python
KODE_ASISTEN = {
    "daffa" : "FDR",
    "dirgi" : "DRG",
    "rufus" : "RFS",
    # tambah mahasiswa baru (max 3 huruf!):
    # "nama" : "KOD",
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
| 100 mahasiswa | ~3.5 jam |
| Smile Detection | Simple MAR (~1ms/frame) |
| Status | ✅ Siap Deploy ke Raspberry Pi |

---

## 💡 Solusi Masalah Umum

| Masalah | Solusi |
|---|---|
| Kamera tidak terbuka (hitam) | Tambahkan `cv2.CAP_DSHOW` + coba index 0 atau 1 |
| Error `/tmp/` di Windows | Sudah difix pakai `tempfile.gettempdir()` |
| Kamera lag | Gunakan `predict_thread.py` bukan `predict.py` |
| Wajah tidak terdeteksi | Pastikan pencahayaan cukup, CLAHE aktif |
| Raspi kepanasan | Naikkan `FRAME_SKIP` di config.py (8 → 10) |
| Shapes not aligned | Model berubah → hapus `.pkl` lama, register ulang |
| JSON not serializable | Pastikan `bool(is_smiling)` bukan numpy bool |
| Smile tidak terdeteksi | Turunkan `SMILE_WHITE_THRESHOLD` di predict.py |
| ImportError SMILE_VOTE | Tambahkan parameter smile di config.py |

---

## 🔗 Deploy ke Raspberry Pi

```bash
# Di Raspberry Pi
git clone https://github.com/Daffa12777/boostify-ml.git
cd boostify-ml
chmod +x ml/setup_raspi.sh
./ml/setup_raspi.sh

# Jalankan
python3 ml/predict_thread.py
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
git commit -m "tambah mahasiswa baru - register"
git push
```

---
# 🔗 Integrasi ML ke Web — Boostify

## 📌 Metode Integrasi yang Dipakai

Boostify menggunakan:

* REST API
* HTTP Request
* JSON
* Express.js API
* Python `requests.post()`
* Frontend `fetch()`

Bukan:

* MQTT
* WebSocket
* SocketIO

---

# 🧠 Alur Integrasi

```txt id="4j9u1y"
Camera
↓
ML Python (Face Recognition)
↓
POST Attendance
↓
Backend Express API
↓
Supabase Database
↓
Frontend Next.js
↓
LiveReport & Recap
```

---

# 🌐 PORT YANG DIPAKAI

| Service          | Port                 |
| ---------------- | -------------------- |
| Backend Express  | 3000                 |
| Frontend Next.js | 3001                 |
| ML Python        | local python process |

---

# 📡 STEP INTEGRASI ML → WEB

## 1. ML Mengirim Data ke Backend

File:

```bash id="l6dj5n"
predict.py
```

Tambahkan:

```python id="7tt8oz"
requests.post(
    "http://localhost:3000/api/uploadfromml",
    json=last_result
)
```

---

## 2. Backend Menyediakan Endpoint

File:

```bash id="hl9qqm"
src/routes/routes.js
```

Tambahkan:

```js id="67wv9u"
router.post("/uploadfromml", uploadAttendanceData);
```

---

## 3. Backend Menyimpan ke Supabase

File:

```bash id="bm29sk"
sendDataController.js
```

Gunakan Prisma:

```js id="p6s6lc"
await prisma.attendance.create({
  data: {
    assistant_code,
    name,
    time
  }
})
```

---

## 4. Frontend Mengambil Data Attendance

File:

```bash id="jxf9ks"
LiveReport.tsx
```

Gunakan:

```ts id="s8gcxk"
fetch("http://localhost:3000/api/attendances")
```

---

## 5. Frontend Menampilkan Data

Data dari backend otomatis muncul di:

* LiveReport
* Recap

---

# 🔐 AUTHENTICATION

Frontend memakai:

```txt id="d9jlwm"
JWT + NextAuth
```

Token dikirim ke backend:

```ts id="rshk8f"
Authorization: `Bearer ${token}`
```

---

# 🛡️ ANTI SPAM

File:

```bash id="y97i8o"
config.py
```

```python id="st3n3m"
COOLDOWN_SEC = 60
```

Tujuan:

* 1 orang tidak spam attendance
* Supabase tetap bersih

---

# 🚀 SAAT DEPLOY

## Local Development

| Service  | URL            |
| -------- | -------------- |
| Backend  | localhost:3000 |
| Frontend | localhost:3001 |

---

## Production

| Service  | URL          |
| -------- | ------------ |
| Backend  | Vercel       |
| Frontend | Vercel       |
| Database | Supabase     |
| ML       | Raspberry Pi |

---

# 📌 FINAL FLOW

```txt id="f3n7mi"
ML Python
↓
POST /api/uploadfromml
↓
Backend Express
↓
Supabase
↓
Frontend Fetch API
↓
LiveReport & Recap
```

---

# ✅ HASIL

✅ Attendance realtime
✅ Data otomatis masuk Supabase
✅ LiveReport realtime
✅ Recap realtime
✅ Anti spam aktif
✅ Siap deploy Raspberry Pi


## 🔗 Integrasi Project

| Komponen | Teknologi | Status |
|---|---|---|
| Frontend | Next.js + TypeScript | ✅ |
| Backend | Express.js + Prisma | ✅ |
| Database | Supabase PostgreSQL | ✅ |
| ML | Python + DeepFace + OpenCV | ✅ |
| Auth | NextAuth + JWT | ✅ |

---

## 👤 Tim


**Repository** — [github.com/Daffa12777/boostify-ml](https://github.com/Daffa12777/boostify-ml)
**Project** — Boostify Face Recognition Attendance System

---

*Boostify ML — Smart Attendance for Smart Campus* 🐝