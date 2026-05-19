import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Kamera gagal dibuka")
    exit()

ret, frame = cap.read()

if ret:
    print("✅ Kamera berhasil")
    print("Ukuran frame:", frame.shape)
else:
    print("❌ Gagal membaca frame")

cap.release()
