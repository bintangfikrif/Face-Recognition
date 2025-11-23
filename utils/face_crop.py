import os
import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np

RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"
TARGET_SIZE = (224, 224)

detector = MTCNN()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def detect_and_crop(image_path):
    """Deteksi wajah pada gambar dan kembalikan hasil crop."""
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)

    if len(results) == 0:
        return None  

    # Ambil wajah dengan confidence terbesar
    results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    x, y, w, h = results[0]['box']

    # Pastikan bounding box aman (tanpa nilai negatif)
    x = max(0, x)
    y = max(0, y)

    cropped = img[y:y+h, x:x+w]

    return cropped

def process_person_folder(person_name):
    """Proses satu folder mahasiswa."""
    raw_folder = os.path.join(RAW_DIR, person_name)
    processed_folder = os.path.join(PROCESSED_DIR, person_name)
    ensure_dir(processed_folder)

    for filename in os.listdir(raw_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(raw_folder, filename)
            crop = detect_and_crop(img_path)

            if crop is None:
                print(f"No face detected in {img_path}")
                continue

            # Resize ke 224x224
            crop = cv2.resize(crop, TARGET_SIZE)
            save_path = os.path.join(processed_folder, filename)

            # Simpan dalam format RGB as-is
            Image.fromarray(crop).save(save_path)

            print(f"Saved processed image → {save_path}")

def process_all():
    """Memproses seluruh folder mahasiswa di data_raw."""
    if not os.path.exists(RAW_DIR):
        print("Folder data_raw tidak ditemukan!")
        return

    persons = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]

    print(f"Memproses {len(persons)} folder mahasiswa...\n")

    for person in persons:
        print(f"Memproses folder: {person}")
        process_person_folder(person)

    print("\nProses selesai! Semua wajah telah dipotong dan disimpan di data_processed.")

if __name__ == "__main__":
    process_all()

