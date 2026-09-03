# 🪸 PANDUAN BELAJAR & PRESENTASI
## Deteksi Fine-Grained Kesehatan Karang dengan YOLOv8

> Dokumen ini dibuat supaya kamu **paham konsepnya**, **bisa menjelaskan ke dosen**,
> dan **siap presentasi**. Tiap bagian punya: (1) apa itu, (2) kenapa dilakukan,
> (3) istilah kunci, (4) kalimat siap-ucap untuk presentasi.

---

## BAGIAN 0 — KONSEP RISET (FONDASI NOVELTY)

### Apa masalahnya?
Karang punya 3 kondisi kesehatan yang perlu dibedakan:

| Kondisi | Penyebab | Tampilan | Status |
|---|---|---|---|
| **healthy** | Alga simbiosis hidup | coklat/hijau/warnai | hidup |
| **bleaching** | Stres suhu → alga keluar | putih merata | masih hidup (jaringan utuh) |
| **white_syndrome (WS)** | bakteri (Vibrio) | lesi putih tajam, kerangka mati | mati di zona lesi |

### Novelty (kebaruan) penelitianmu
Kebanyakan riset menggabungkan bleaching & white_syndrome jadi satu kelas "putih".
Kamu **memisahkan keduanya** = *fine-grained classification* (klasifikasi tingkat halus).

### Kenapa penting? (untuk BRIN / dosen)
- Bleaching **reversibel** (bisa pulih kalau suhu normal).
- White syndrome **penyakit mematikan** (butuh penanganan berbeda).
- Membedakan keduanya punya implikasi ekologi & konservasi yang berbeda.

🎤 **Kalimat presentasi:**
> "Mayoritas literatur memperlakukan bleaching dan white syndrome sebagai satu
> kategori 'putih'. Penelitian ini memisahkan keduanya karena implikasi
> ekologisnya berbeda — bleaching reversibel, sementara white syndrome adalah
> penyakit mematikan. Inilah novelty fine-grained kami."

---

## BAGIAN 1 — SETUP ENVIRONMENT (Mac + VS Code)

### Konsep
- **Virtual environment (venv)** = "ruangan terpisah" untuk paket Python agar
  tidak bentrok dengan Python sistem. Aktifkan dengan `source coral_env/bin/activate`.
- **MPS** = Metal Performance Shaders, GPU bawaan Apple Silicon. PyTorch pakai
  GPU Mac lewat `device="mps"` (setara `cuda` di GPU NVIDIA).
- **requirements.txt** = daftar dependency (bisa di-install ulang di komputer lain).

### Dependency kunci
- `ultralytics` → library YOLOv8
- `label-studio` → tool anotasi manual
- `opencv-python`, `matplotlib`, `pandas`, `scikit-learn` → analisis & visualisasi

🎤 **Kalimat presentasi:**
> "Kami menggunakan Mac Apple Silicon dengan akselerasi GPU via MPS. Dependensi
> dikelola dalam virtual environment agar reproduktif, dan didokumentasikan
> dalam requirements.txt."

### Verifikasi MPS (wajib sebelum training)
```python
import torch
print(torch.backends.mps.is_available())  # harus True
```
Jika `False`, training tetap jalan tapi pakai CPU (lambat).

---

## BAGIAN 2 — STRUKTUR PROYEK

### Konsep
Memisahkan **gambar (images)** dan **label (annotations)**, lalu dibagi
**train / val / test**. File `coral.yaml` adalah "peta" yang dibaca YOLO.

```
data/
 ├── images/{train,val,test}/
 ├── labels/{train,val,test}/
 └── coral.yaml          ← definisi kelas & lokasi data
```

### `coral.yaml` — definisi kelas
```yaml
names:
  0: healthy
  1: bleaching
  2: white_syndrome
```
Angka 0/1/2 = ID kelas yang dipakai di file label `.txt`.

🎤 **Kalimat presentasi:**
> "Struktur proyek memisahkan gambar dan label, lalu membaginya menjadi
> train, validation, dan test. File coral.yaml mendefinisikan tiga kelas
> dan lokasi datanya — ini antarmuka antara dataset dan model YOLO."

---

## BAGIAN 3 — ANOTASI DENGAN LABEL STUDIO

### Konsep
Model belajar dari contoh yang **kita tandai manual**. Kita gambar **polygon**
(multigon mengikuti bentuk karang) di tiap koloni, lalu beri label kondisinya.

### Protokol anotasi (INI INTINYA — bisa ditanya dosen)
| Kelas | Cara anotasi |
|---|---|
| healthy | polygon ikut kontur koloni berwarna/utuh |
| bleaching | polygon SELURUH area putih merata (jaringan masih nutup kerangka) |
| white_syndrome | polygon HANYA zona lesi (batas jaringan hidup ↔ kerangka mati), **jangan** ikut anotasi bagian sehat |

Aturan emas:
- WS = batas tajam → anotasi presisi ikut tepi lesi.
- Bleaching = gradasi halus → boleh sedikit longgar.
- 1 gambar ada koloni campuran → buat polygon **terpisah** per kondisi.
- Target: 150–300 gambar per kelas.

🎤 **Kalimat presentasi:**
> "Anotasi dilakukan dengan polygon pada Label Studio. Protokol kami membedakan
> secara tegas: bleaching dianotasi pada seluruh area putih merata, sedangkan
> white syndrome hanya pada zona lesi tajam. Hal ini krusial agar model belajar
> ciri pembeda, bukan sekadar 'warna putih'."

---

## BAGIAN 4 — KONVERSI ANOTASI → FORMAT YOLO

### Konsep
Label Studio export dalam **JSON** (koordinat dalam %). YOLO butuh **file `.txt`**
per gambar dengan format:
- **Segmentation**: `class_id x1 y1 x2 y2 ...` (koordinat piksel)
- **Bounding box**: `class_id x_center y_center width height` (nilai 0–1)

Script `convert_labels.py`:
1. Baca JSON export.
2. Ambil nama file gambar & polygon-nya.
3. Ubah koordinat % → piksel (mode `seg`) atau → bbox (mode `bbox`).
4. Tulis ke file `.txt` per gambar.

🎤 **Kalimat presentasi:**
> "Kami mengonversi anotasi JSON dari Label Studio ke format YOLO. Koordinat
> polygon dalam persen dikalikan ukuran piksel gambar, lalu ditulis per file
> teks sesuai konvensi YOLO — baik sebagai segmentation maupun bounding box."

---

## BAGIAN 5 — SPLIT DATASET STRATIFIED (70/20/10)

### Konsep
- **Train (70%)** = belajar.
- **Val (20%)** = pantau saat training, atur hyperparameter.
- **Test (10%)** = evaluasi akhir (tidak pernah dilihat saat training).
- **Stratified** = proporsi tiap kelas dijaga sama di setiap split, supaya
  tidak ada split yang kekurangan satu kelas (mis. WS langka).

Script mengelompokkan gambar berdasar **kombinasi kelas** yang ada, lalu membagi
proporsional → mencegah bias/imbalance antar split.

🎤 **Kalimat presentasi:**
> "Dataset dibagi 70/20/10 secara stratified per kelas. Stratifikasi menjamin
> representasi ketiga kelas seimbang di tiap split, sehingga evaluasi tidak bias
> terhadap kelas yang jarang muncul seperti white syndrome."

---

## BAGIAN 6 — TRAINING (Mac / MPS)

### Konsep YOLOv8-seg
- **YOLO** = You Only Look Once, model deteksi objek real-time.
- **-seg** = versi segmentation (keluar polygon, bukan cuma kotak).
- **yolov8s** = ukuran "small". n=mini (cepat), s=small, m=medium (akurat).

### Hyperparameter penting & alasannya
| Param | Nilai | Kenapa |
|---|---|---|
| `imgsz=1024` | tinggi | lesi WS kecil → butuh resolusi tinggi |
| `batch=4` | kecil | RAM GPU Mac terbatas |
| `hsv_h=0.0` | **0** | **jangan ubah hue/warna** — warna = pembeda BS vs WS |
| `hsv_s=0.3, hsv_v=0.4` | sedang | ubah saturasi/terang aman |
| `erasing=0.0` | 0 | jangan hapus area → lesi bisa hilang |
| `mosaic=1.0, mixup=0.15` | on | data augmentation biar model kuat |
| `cos_lr=True` | on | learning rate turun halus |
| `patience=30` | — | berhenti otomatis kalau tak membaik |

### Strategi bertahap (hemat waktu)
1. **Sanity check** (5 epoch, imgsz 320) → pastikan pipeline jalan.
2. **Eksperimen** (50 epoch, imgsz 640) → cari setting.
3. **Final** (150 epoch, imgsz 1024) → jalan semalam.

🎤 **Kalimat presentasi:**
> "Kami melatih YOLOv8-seg dengan resolusi 1024 px karena lesi white syndrome
> berukuran kecil. Poin krusial: kami mematikan augmentasi hue (hsv_h=0) karena
> warna adalah fitur pembeda utama antara bleaching dan white syndrome. Pipeline
> diuji bertahap — dari sanity check 5 epoch hingga training final 150 epoch."

---

## BAGIAN 7 — EVALUASI (BUKTI NOVELTY)

### Konsep metrik
- **mAP@50** = mean Average Precision, IoU threshold 50% (longgar).
- **mAP@50-95** = rata-rata mAP dari IoU 50%–95% (ketat).
- **AP per class** = akurasi tiap kelas (healthy/bleaching/WS).
- **Confusion matrix** = tabel "prediksi vs aslinya" → tunjukkan apakah model
  **tidak mencampur** bleaching ↔ white_syndrome.

### Yang jadi bukti novelty
Confusion matrix dengan kesalahan **bleaching↔WS rendah** = model benar-benar
bisa membedakan dua kondisi "sama-sama putih".

🎤 **Kalimat presentasi:**
> "Evaluasi dilakukan di test set yang terpisah. Metrik utama adalah mAP dan
> AP per kelas. Bukti novelty kami adalah confusion matrix: tingkat kesalahan
> klasifikasi antara bleaching dan white syndrome rendah, menunjukkan model
> mampu membedakan dua kondisi yang secara visual serupa."

---

## BAGIAN 8 — INFERENCE / PREDIKSI

### Konsep
Model sudah jadi → kasih gambar/video baru, model keluar prediksi + confidence.
`conf=0.35` = ambang keyakinan minimal 35% untuk menampilkan deteksi.

🎤 **Kalimat presentasi:**
> "Model yang telah dilatih dapat memproses gambar atau video baru secara
> otomatis, menghasilkan deteksi beserta tingkat keyakinannya — memungkinkan
> pemantauan kesehatan karang otomatis di lapangan."

---

## BAGIAN 9 — NOTEBOOK ANALISIS

### Konsep
- **Kurva training** (precision/recall/loss) → lihat apakah model stabil/belajar.
- **Distribusi kelas** → cek imbalance (apakah satu kelas terlalu sedikit).

🎤 **Kalimat presentasi:**
> "Kami menganalisis kurva pelatihan dan distribusi kelas melalui notebook
> Jupyter untuk memastikan model konvergen dan dataset seimbang."

---

## 📋 GLOSARIUM ISTILAH (hafal ini)

| Istilah | Arti singkat |
|---|---|
| Fine-grained | Klasifikasi detail/sangat spesifik (bukan cuma kasar) |
| Annotation | Penandaan manual objek pada gambar |
| Polygon | Garis multigon mengikuti bentuk objek |
| Segmentation | Membagi gambar jadi region berlabel |
| Bounding box | Kotak pembatas objek |
| mAP | Rata-rata akurasi deteksi |
| IoU | Tingkat tumpang-tindih prediksi vs aslinya |
| Augmentation | Mengubah gambar (flip/rotasi) untuk tambah variasi |
| Stratified split | Bagi data sambil jaga proporsi kelas |
| MPS / GPU | Akselerasi komputasi (Mac / kartu grafis) |
| Epoch | 1x model lihat seluruh data training |
| Batch | Seberapa banyak gambar diproses sekaligus |

---

## ❓ SIMULASI PERTANYAAN DOSEN & JAWABAN

**Q: Kenapa Anda tidak gabung saja bleaching & white syndrome?**
A: Karena implikasi ekologisnya berbeda — bleaching reversibel, WS penyakit
mematikan. Memisahkan keduanya adalah kontribusi utama penelitian ini.

**Q: Kenapa imgsz tinggi (1024)?**
A: Lesi white syndrome berukuran kecil; resolusi rendah membuatnya hilang detail.

**Q: Kenapa hsv_h = 0 (tidak ubah warna)?**
A: Warna adalah fitur pembeda krusial antara bleaching (putih) dan WS (lesi).
Mengubah hue akan merusak sinyal tersebut.

**Q: Apa bedanya train, val, test?**
A: Train untuk belajar, val untuk tuning saat training, test untuk evaluasi akhir
yang belum pernah dilihat — uji generalisasi sesungguhnya.

**Q: Bagaimana bukti model Anda berhasil?**
A: Confusion matrix menunjukkan kesalahan bleaching↔WS rendah, dan AP per kelas
tinggi di test set.

**Q: Kenapa YOLO bukan CNN biasa?**
A: YOLO deteksi + segmentasi sekaligus real-time, cocok untuk pemantauan otomatis.

---

## 🎤 STRUKTUR PRESENTASI (10–15 menit)

1. **Latar belakang** (2 mnt): masalah karang, 3 kondisi, novelty memisahkan BS/WS.
2. **Metodologi** (4 mnt): dataset → anotasi → konversi → split → training.
3. **Detail kunci** (3 mnt): imgsz tinggi, hsv_h=0, stratified split.
4. **Evaluasi** (3 mnt): mAP, confusion matrix, bukti BS≠WS.
5. **Demo & kesimpulan** (2 mnt): prediksi gambar nyata, kontribusi penelitian.

---

## ✅ CHECKLIST SEBELUM PRESENTASI
- [ ] Paham perbedaan healthy / bleaching / white_syndrome
- [ ] Bisa jelaskan kenapa hsv_h = 0
- [ ] Bisa jelaskan fungsi train/val/test
- [ ] Paham arti mAP & confusion matrix
- [ ] Siap tunjukkan confusion matrix sebagai bukti novelty
- [ ] Hafal glosarium & jawaban pertanyaan dosen
