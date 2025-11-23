# Face Recognition for Student Attendance — Deep Learning (IF25-40401) Project

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
project-face-recognition/
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

