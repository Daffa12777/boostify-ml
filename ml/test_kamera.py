import cv2

print("Mencari kamera...")

for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"✅ Kamera ditemukan di index: {i}")
        cap.release()
    else:
        print(f"❌ Index {i} tidak ada kamera")