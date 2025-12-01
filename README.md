# Deep Learning (IF25-40401) Project
# NeoFace - Face Recognition for Student Attendance 


Proyek ini dibuat sebagai bagian dari Tugas Besar Mata Kuliah Pembelajaran Mendalam (IF25-40401) di Program Studi Teknik Informatika, Institut Teknologi Sumatera.  
Topik utama tugas adalah membangun sistem presensi mahasiswa berbasis *Face Recognition* menggunakan model *Deep Learning end-to-end*.

---

## Daftar Anggota Kelompok

| Nama | NIM | Profil GitHub |
|------|------|----------------|
| Bintang Fikri Fauzan | 122140008 | https://github.com/bintangfikrif |
| Ferdana Al Hakim | 122140012 | https://github.com/luciferdana |
| Zidan Raihan | 122140100 | https://github.com/zidbytes |


---

## Struktur Direktori Proyek

```
Face-Recognition/
│
├── data_raw/ # Dataset asli 
├── data_processed/ # Dataset hasil preprocessing
├── notebooks/ # Notebook training 
├── models/ # Model hasil training
├── app/ # Aplikasi demo 
├── utils/ # Helper functions
├── requirements.txt # Dependency library
└── README.md # Dokumentasi project
```

## Dataset

Dataset diperoleh dari foto mahasiswa yang diunggah melalui Google Form.  
Setiap mahasiswa mengunggah minimal 5 foto dengan ketentuan:
- Wajah terlihat jelas  
- Tidak tertutup masker/kacamata hitam  
- Pose bebas  
- Background bebas  

## Preprocessing

Data raw yang sudah dikumpulkan dalam satu folder `data_raw/` selanjutnya dilakukan proses cropping wajah menggunakan MediaPipe dan disimpan dalam folder `data_processed/`. Proses ini dilakukan dengan menggunakan script `utils/face_crop.py`.

## Model yang digunakan

Pada project ini, digunakan model **FaceNet** sebagai model utama untuk klasifikasi wajah. Dengan kombinasi hyperpamater sebagai berikut.

### Hyperparameter
- Batch Size: 32
- Learning Rate: 1e-4
- Epoch: 50
- Image Size: 160x160
- Optimizer: Adam
- Weight Decay: 1e-4
- Scheduler: StepLR
- Num Workers: 2

### Hasil Training

Hasil traning model yang telah dilakukan adalah sebagai berikut

**Training Loss & Accuracy**
![FaceNet Training Loss & Accuracy](checkpoints\facenet_run\training_history.png)

Training loss turun sangat cepat hingga hampir nol, menunjukkan model belajar sangat kuat pada data training. Namun, validation loss hanya menurun di awal lalu stagnan dan berfluktuasi, menandakan model tidak mampu mempertahankan performa di data baru. Perbedaan besar ini menunjukkan adanya overfitting.

Training accuracy meningkat hingga hampir 100%, tetapi validation accuracy justru berfluktuasi dan tidak stabil. Hal ini menunjukkan model sangat fit terhadap data training, namun kurang mampu melakukan generalisasi. Pola ini kembali menguatkan indikasi overfitting pada model.

**Prediction Samples**
![Swin Transformer Prediction Samples](checkpoints\swin_run_3(best)\prediction_samples.png)

Dari 5 sampel prediksi, model berhasil menebak 3 dari 5 sampel. Menandakan bahwa model belum maksimal dalam mengenali wajah mahasiswa.

Hasil Uji coba pada Data Test:
- Top-1 Accuracy: 73.91%
- Top-5 Accuracy: 86.96%

### DeiT Small

**Training Loss & Accuracy**
![DeiT Small Training Loss & Accuracy](checkpoints\deit_run\training_history.png)

Training loss menurun dengan sangat cepat hingga hampir nol, menunjukkan model belajar dengan baik pada data training. Sementara itu, validation loss hanya menurun pada awal training lalu bergerak fluktuatif di kisaran yang lebih tinggi, menandakan performa model pada data baru tidak meningkat sebaik di data training. Perbedaan drastis antara keduanya kembali mengindikasikan overfitting.

Training accuracy naik tajam hingga mendekati 100% dalam waktu singkat, tetapi validation accuracy hanya naik di awal dan kemudian berfluktuasi tanpa tren yang stabil. Pola ini menunjukkan bahwa meskipun model sangat akurat pada data training, kemampuannya melakukan generalisasi masih lemah. Ketidakstabilan pada validation accuracy memperkuat kesimpulan bahwa model mengalami overfitting.

**Prediction Samples**
![DeiT Small Prediction Samples](checkpoints\deit_run\prediction_samples.png)

Dari 5 sampel prediksi, model berhasil menebak 3 dari 5 sampel. Menandakan bahwa model belum maksimal dalam mengenali wajah mahasiswa.

Hasil Uji coba pada Data Test:
- Top-1 Accuracy: 56.52%
- Top-5 Accuracy: 69.57%

## Aplikasi Demo

### Deployment

Aplikasi demo dapat dijalankan dan di deploy dengan menggunakan Streamlit. 

### Alur Aplikasi
1. **Inisiasi & Pemuatan Model**. Saat aplikasi dijalankan, ia memuat model Deep Learning yang telah dilatih dari folder `models/`. Aplikasi juga memuat mapping kelas yang diperlukan untuk mengetahui nama mahasiswa mana yang sesuai dengan hasil klasifikasi model. Aplikasi didesain agar pengguna dapat memilih model .pth yang berbeda secara dinamis melalui sidebar.

2. **Input Pengguna**. Pengguna mengunggah sebuah foto ke dalam antarmuka aplikasi Streamlit.

3. **Deteksi Wajah**. Setelah foto diunggah, aplikasi akan menggunakan MTCNN (Multi-task Cascaded Convolutional Networks) untuk mendeteksi wajah yang ada pada gambar tersebut.

4. **Klasifikasi (Inferensi)**. Bagian wajah yang terdeteksi kemudian diteruskan ke model Deep Learning yang telah dimuat (sesuai pilihan pengguna). Model melakukan inferensi klasifikasi wajah untuk menentukan identitas mahasiswa yang bersangkutan.

5. **Output Hasil**. Aplikasi akan menampilkan hasil prediksi. Hasil tersebut mencakup menampilkan bounding box di sekitar wajah yang terdeteksi. Aplikasi juga menampilkan label prediksi, yang berisi Nama Mahasiswa beserta nilai Confidence (tingkat keyakinan model).

## Kesimpulan
