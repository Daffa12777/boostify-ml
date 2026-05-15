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

````md
# BOOSTIFY — AI Attendance System

Boostify adalah sistem absensi berbasis AI Face Recognition + Smile Detection yang terintegrasi dengan:

- Frontend Next.js
- Backend Express.js
- Prisma ORM
- Supabase PostgreSQL
- Machine Learning Python (DeepFace + OpenCV)

---

# FITUR

✅ Face Recognition  
✅ Smile Detection  
✅ JWT Authentication  
✅ NextAuth Login  
✅ Live Attendance  
✅ Recap Attendance  
✅ Supabase Integration  
✅ Cooldown Anti Spam  
✅ Raspberry Pi Ready  

---

# TECH STACK

## Frontend
- Next.js
- React
- TypeScript
- NextAuth

## Backend
- Express.js
- Prisma ORM
- JWT
- bcryptjs

## Machine Learning
- Python
- DeepFace
- OpenCV
- GhostFaceNet

## Database
- Supabase PostgreSQL

---

# FLOW SISTEM

Camera ML
↓
Face Recognition
↓
Smile Detection
↓
Send Attendance Data
↓
Backend Express API
↓
Prisma ORM
↓
Supabase Database
↓
LiveReport & Recap

---

# STRUKTUR PROJECT

## Frontend
/pages
/components
/styles

## Backend
/src
/routes
/features

## ML
predict.py
config.py
train.py

---

# LOGIN AUTHENTICATION

Menggunakan:

- NextAuth
- JWT Token
- Credentials Provider

## File penting
```bash
pages/api/auth/[...nextauth].js
````

---

# SESSION FLOW

## Login berhasil

```js
return {
  id: user.payload.id,
  name: user.payload.name,
  assistantCode: user.payload.assisstant_code,
  token: user.token,
};
```

## JWT callback

```js
token.id = user.id;
token.name = user.name;
token.assistantCode = user.assistantCode;
token.token = user.token;
```

## Session callback

```js
session.user.id = token.id;
session.user.name = token.name;
session.user.assistantCode = token.assistantCode;
session.user.token = token.token;
```

---

# MIDDLEWARE PROTECTION

Protected routes:

```js
const protectedRoutes = [
  '/HomePage',
  '/Profile',
  '/Recap',
  '/LiveReport'
];
```

Middleware menggunakan:

```js
getToken({ req, secret })
```

---

# CORS CONFIGURATION

Backend:

```js
app.use(cors(corsOptions));
```

Allowed origin:

```js
origin: [
  "http://localhost:3000",
  "http://localhost:3001",
  "https://boostify-fe.vercel.app"
]
```

---

# LIVE REPORT API

Endpoint backend:

```js
router.get('/attendances')
```

Frontend fetch:

```ts
/api/attendances
```

---

# RECAP API

Endpoint:

```ts
/api/recap?page=${currentPage}
```

Authorization:

```ts
Authorization: `Bearer ${token}`
```

---

# ML FACE RECOGNITION

Model:

```python
MODEL_BACKEND = "GhostFaceNet"
```

Detector:

```python
DETECTOR_BACKEND = "opencv"
```

---

# SMILE DETECTION

Menggunakan:

```python
haarcascade_smile.xml
```

---

# ANTI SPAM COOLDOWN

Masalah:
1 wajah upload berkali-kali.

Solusi:

```python
last_upload_time = {}
```

Cooldown:

```python
COOLDOWN_SEC = 60
```

Logic:

```python
if current_time - last_upload_time[nama] > COOLDOWN_SEC:
```

Hasil:

* 1 orang hanya upload 1x tiap 60 detik
* Supabase tidak spam
* Recap lebih rapi

---

# INTEGRASI ML → BACKEND

## ML kirim data

```python
requests.post(
    "http://localhost:3000/api/uploadfromml",
    json=last_result
)
```

## Backend route

```js
router.post("/uploadfromml", uploadAttendanceData);
```

---

# FORMAT DATA ATTENDANCE

```json
{
  "assisstant_code": "FDR",
  "name": "daffa",
  "time": "2026-05-15T12:00:00Z",
  "uuid": "xxxx-xxxx",
  "formattedTime": "Friday, May 15, 2026",
  "is_smiling": true
}
```

---

# FILE PENTING

## Frontend

```bash
pages/api/auth/[...nextauth].js
pages/SignIn.tsx
pages/LiveReport.tsx
pages/Recap.tsx
```

## Backend

```bash
src/index.js
src/routes/routes.js
src/features/auth/services/loginService.js
```

## ML

```bash
predict.py
config.py
train.py
```

---

# CARA MENJALANKAN

## Frontend

```bash
npm install
npm run dev
```

## Backend

```bash
npm install
npm run dev
```

## ML

```bash
python predict.py
```

---

# STATUS PROJECT

✅ Login berhasil
✅ JWT berhasil
✅ Middleware berhasil
✅ Session berhasil
✅ LiveReport berhasil
✅ Recap berhasil
✅ Supabase berhasil
✅ ML berhasil
✅ Smile Detection berhasil
✅ Anti Spam berhasil
✅ Realtime Attendance berhasil

---

# AUTHOR

Boostify Team

```
```


## 👤 Tim

**Machine Learning** — Muhammad Daffa  
**Repository** — [github.com/Daffa12777/boostify-ml](https://github.com/Daffa12777/boostify-ml)  
**Project** — Boostify Face Recognition Attendance System

---

*Boostify ML — Smart Attendance for Smart Campus* 🐝