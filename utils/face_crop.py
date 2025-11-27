import os
import cv2
import mediapipe as mp # Ganti MTCNN dengan MediaPipe
from PIL import Image
import numpy as np

# Dapatkan path root project (satu level di atas folder utils)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data_raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "padded_data_processed")
TARGET_SIZE = (224, 224)

# --- Inisialisasi MediaPipe ---
mp_face_detection = mp.solutions.face_detection
# model_selection=1 cocok untuk full range (jarak jauh/dekat), 0 untuk jarak dekat (selfie)
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def detect_and_crop(image_path):
    """Deteksi wajah pada gambar dan kembalikan hasil crop menggunakan MediaPipe."""
    
    # --- PROSES LOAD GAMBAR (Sama seperti sebelumnya) ---
    stream = open(image_path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    stream.close()

    if img is None:
        print(f"❌ Gagal membaca gambar: {image_path}")
        return None

    # Normalisasi dtype
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    # Convert ke RGB
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize jika gambar terlalu besar
    h_orig, w_orig = img.shape[:2]
    max_dim = 1024
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        img = cv2.resize(img, (new_w, new_h))
    
    # Update dimensi setelah resize (penting untuk kalkulasi MediaPipe)
    h, w = img.shape[:2]

    # --- PROSES MEDIAPIPE ---
    # MediaPipe butuh input RGB (img sudah RGB di tahap ini)
    results = detector.process(img)

    if not results.detections:
        return None  

    # Ambil wajah dengan confidence (score) terbesar
    # MediaPipe mengembalikan list detections, kita sort berdasarkan score[0]
    best_detection = sorted(results.detections, key=lambda x: x.score[0], reverse=True)[0]
    
    # Ambil Bounding Box Relatif (0.0 - 1.0)
    bbox = best_detection.location_data.relative_bounding_box
    
    # Konversi ke Pixel (Absolut)
    # MediaPipe kadang mengembalikan nilai < 0 atau > 1, jadi kita clamp manual nanti
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    w_box = int(bbox.width * w)
    h_box = int(bbox.height * h)

    # Pastikan koordinat awal tidak negatif (Clean up hasil konversi)
    x = max(0, x)
    y = max(0, y)

    # --- Tambahkan Padding (Logika Lama Dipertahankan) ---
    padding = 0.2  # 20% padding
    
    pad_w = int(w_box * padding)
    pad_h = int(h_box * padding)
    
    # Update koordinat dengan padding
    x_new = max(0, x - pad_w)
    y_new = max(0, y - pad_h)
    
    # Pastikan width/height baru tidak keluar batas gambar
    w_new = min(w - x_new, w_box + 2 * pad_w)
    h_new = min(h - y_new, h_box + 2 * pad_h)
    
    # Lakukan Cropping
    cropped = img[y_new:y_new+h_new, x_new:x_new+w_new]

    return cropped

def process_person_folder(person_name):
    """Proses satu folder mahasiswa."""
    raw_folder = os.path.join(RAW_DIR, person_name)
    processed_folder = os.path.join(PROCESSED_DIR, person_name)
    ensure_dir(processed_folder)

    # Cek apakah folder raw ada
    if not os.path.exists(raw_folder):
        print(f"Skipping {person_name}, folder not found.")
        return

    for filename in os.listdir(raw_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp", ".heic")):
            img_path = os.path.join(raw_folder, filename)
            crop = detect_and_crop(img_path)

            if crop is None:
                print(f"No face detected in {filename}")
                continue

            # Resize ke 224x224 (TARGET_SIZE)
            try:
                crop = cv2.resize(crop, TARGET_SIZE)
                save_path = os.path.join(processed_folder, filename)

                # Simpan dalam format RGB as-is
                Image.fromarray(crop).save(save_path)
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")

def process_all():
    """Memproses seluruh folder mahasiswa di data_raw."""
    if not os.path.exists(RAW_DIR):
        print("Folder data_raw tidak ditemukan!")
        return

    persons = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]

    print(f"Memproses {len(persons)} folder mahasiswa...\n")

    for person in persons:
        print(f"--- Memproses folder: {person} ---")
        process_person_folder(person)

    print("\nProses selesai! Semua wajah telah dipotong dan disimpan.")

if __name__ == "__main__":
    process_all()
    # Penting: Tutup detector setelah selesai semua proses
    detector.close()