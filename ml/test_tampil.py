import cv2

# Coba ganti angka 0 atau 1
cap = cv2.VideoCapture(1)

print("Kamera terbuka. Tekan Q untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal baca frame")
        break

    cv2.imshow("Test Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()