# 🪸 MASTER PLAYBOOK v2.0 — Deteksi Fine-grained Kesehatan Karang dengan YOLOv8

> **Panduan lengkap dari nol sampai evaluasi** (versi revisi sesuai kondisi dataset-mu saat ini)

---

## 📋 KONDISI TERKINI KAMU (yang jadi dasar pembaruan)

| Aspek | Status |
|-------|--------|
| Hardware | Mac Apple Silicon (MPS), VS Code ✅ |
| Data mentah | 17.038 gambar di `/Users/mac/Desktop/BRIN/Kumpulan Dataset Coral/dataset_coral` (flat, nama file ber-prefix sumber) ✅ |
| Format label | Klasifikasi image-level saja — **tidak ada** koordinat box/polygon ❌ |
| Label tersedia | bleached, healthy, dead, unhealthy — **tidak ada** white_syndrome eksplisit ❌ |
| Masalah terdeteksi | dataset5 ter-augmentasi (risiko leakage); kemungkinan duplikat lintas dataset ⚠️ |

> 💡 **Temuan penting:** ukuran dataset5 testing (485 + 438) = persis sama dengan dataset1 → hampir pasti dataset5 adalah versi augmentasi dari dataset1. Ini memperkuat keputusan mengeksklusikan dataset5.

---

## 🔑 Strategi Dua Tahap (inti pembaruan)

```text
TAHAP A — Baseline Klasifikasi          TAHAP B — Novelty Fine-grained
Ribuan gambar gratis dari               Subset ±500–1000 gambar yang KAUMANUAT
dataset publik (tanpa anotasi             anotasi POLYGON presisi di Label Studio
spasial)
        ↓                                                       ↓
Model klasifikasi: Healthy/              YOLOv8-seg: healthy / bleaching /
Bleached/Dead + hunting WS               white_syndrome ← FOKUS RISETMU
```

---

## BAGIAN 1 — Environment Setup (Mac + VS Code)

```bash
mkdir -p ~/Desktop/BRIN/coral-detection && cd ~/Desktop/BRIN/coral-detection
python3 -m venv coral_env
source coral_env/bin/activate

pip install ultralytics label-studio opencv-python pandas matplotlib \
            scikit-learn seaborn Pillow
pip freeze > requirements.txt
```

Verifikasi GPU Mac:

```python
import torch
print(torch.backends.mps.is_available())  # harus True
```

VS Code: install ekstensi Python, Pylance, Jupyter → `Cmd+Shift+P` → *Select Interpreter* → `coral_env`. Buat `.vscode/settings.json`:

```json
{"python.defaultInterpreterPath": "${workspaceFolder}/coral_env/bin/python"}
```

---

## BAGIAN 2 — Struktur Proyek Final (UPDATED)

```text
Desktop/BRIN/
├── Kumpulan Dataset Coral/
│   ├── dataset1..7/              ← MENTAH ASLI, JANGAN DISENTUH
│   └── dataset_coral/            ← FLAT COPY 17k (bahan audit/dedup)
└── coral-detection/
    ├── .vscode/settings.json
    ├── .gitignore
    ├── notebooks/
    │   ├── 00_inventory.ipynb
    │   ├── 01_dedup_audit.ipynb
    │   ├── 02_visual_check.ipynb     ⬅️ HUNTING WHITE SYNDROME
    │   └── 03_analyze_results.ipynb
    ├── harmonize_v2.py           ⬅️ BARU: parsing dari folder flat
    ├── cek_duplikat.py
    ├── convert_labels.py         (sudah dibuatkan sebelumnya)
    ├── split_dataset.py          (sudah dibuatkan sebelumnya)
    ├── train_cls_a.py            ⬅️ BARU: training klasifikasi Tahap A
    ├── train.py                  (YOLOv8-seg Tahap B)
    ├── val.py
    ├── predict.py
    ├── manifest_tahap_a.csv
    ├── data_tahap_a/{healthy,bleached,dead}/   ← hasil harmonisasi
    └── data/
        ├── images/{train,val,test}/
        ├── labels/{train,val,test}/
        └── coral.yaml
```

### `.gitignore`

```gitignore
coral_env/
runs/
data/
data_tahap_a/
*.pt
.DS_Store
__pycache__/
.ipynb_checkpoints/
```

> Folder `dataset_coral` (692 MB) biarkan di luar proyek — tidak ikut Git.

---

## BAGIAN 3 — 🔍 AUDIT DATA & DEDUPLIKASI (baru)

Jalankan pada folder flat-mu (script `cek_duplikat.py` dengan dHash sudah dibuatkan sebelumnya):

```bash
cd coral-detection
python cek_duplikat.py   # arahkan ke "../Kumpulan Dataset Coral/dataset_coral"
```

Yang kamu dapatkan:
- `duplikat_laporan.txt` — daftar kelompok gambar kembar
- Bukti ilmiah untuk laporan: *"Ditemukan N kelompok duplikat lintas-dataset, dikeluarkan dari test set untuk mencegah leakage"*

> **Ekspektasi hasil:** duplikat pasti banyak antara dataset1 ↔ dataset5 dan mungkin dataset2 ↔ dataset5. Karena dataset5 sudah dieksklusi, sisanya aman.

---

## BAGIAN 4 — HARMONISASI v2: Parsing Label dari Nama File (baru)

Folder flat-mu punya pola nama: `datasetN_subfolder_namaasli.ext`. Script ini membaca tanpa perlu menyentuh folder asli:

```python
# harmonize_v2.py
"""
Ambil gambar dari folder FLAT 'dataset_coral' -> klasifikasikan
ke data_tahap_a/{healthy, bleached, dead, REVIEW} berdasarkan prefix nama file.
dataset5 otomatis DIEKSKLUSI.
"""
import re
import shutil
from pathlib import Path
import pandas as pd

SRC = Path("/Users/mac/Desktop/BRIN/Kumpulan Dataset Coral/dataset_coral")
OUT = Path("data_tahap_a")
OUT.mkdir(exist_ok=True)

def classify(fname: str):
    """Return (label, split_info) dari nama file ber-prefix."""
    name = fname.lower()
    if name.startswith("dataset5_"):
        return None, None                      # ⛔ EXCLUDE (augmented)

    if "_coral_bl" in name:      return "bleached", None
    if "_bleached" in name:      return "bleached", None
    if "_dead" in name:          return "dead",     None

    if "_healthy" in name:       return "healthy",  None
    if "_unbleached" in name:    return "healthy",  None
    if "_coral_" in name:        return "healthy",  None   # CORAL polos ds2

    if "_unhealthy" in name:     return "REVIEW",   None   # ⚠️ definisi ambigu
    return None, None                                       # tak terkenali

rows = []
for f in sorted(SRC.iterdir()):
    if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    label, _ = classify(f.name)
    if label:
        dest = OUT / label
        dest.mkdir(exist_ok=True)
        shutil.copy2(f, dest / f.name)
        rows.append({"filename": f.name, "label": label})
    else:
        rows.append({"filename": f.name, "label": "EXCLUDED_OR_UNKNOWN"})

df = pd.DataFrame(rows)
df.to_csv("manifest_tahap_a.csv", index=False)
print(df["label"].value_counts())
```

```bash
python harmonize_v2.py
```

### Tabel mapping (untuk laporanmu)

| Pola nama file | Label standar | Catatan |
|----------------|---------------|---------|
| `*_bleached*`, `*_CORAL_BL*`, `dataset6_Bleached_*` | **bleached** | konsisten antar-sumber |
| `*_healthy*`, `*_Unbleached_*`, `*_CORAL_*` (ds2 polos) | **healthy** | |
| `dataset3_Dead_*`, `dataset7_dead_*` | **dead** | kandidat sumber WS 🔍 |
| `dataset7_unhealthy_*` | **REVIEW** | definisi ambigu → cek visual wajib |
| `dataset5_*` | **di-exclude** | augmentasi → leakage risk |

---

## BAGIAN 5 — 🎯 VERIFIKASI VISUAL & HUNTING WHITE SYNDROME

Ini tahap yang **MENENTUKAN novelty-mu**. Buka notebook `02_visual_check.ipynb`:

```python
import matplotlib.pyplot as plt, random
from PIL import Image
from pathlib import Path

def show_grid(folder, n=20):
    files = list(Path(folder).glob("*"))
    sample = random.sample(files, min(n, len(files)))
    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    for ax, f in zip(axes.flat, sample):
        ax.imshow(Image.open(f)); ax.set_title(f.name[:34], fontsize=6); ax.axis("off")
    plt.suptitle(folder); plt.tight_layout(); plt.show()

show_grid("data_tahap_a/dead")        # ⬅️ PRIORITAS 1
show_grid("data_tahap_a/REVIEW")      # ⬅️ PRIORITAS 2
show_grid("data_tahap_a/bleached")
show_grid("data_tahap_a/healthy")
```

### Apa yang dicari dalam kelas Dead dan unhealthy:

| Jika melihat… | Berarti |
|---------------|---------|
| Garis putih tajam + jaringan hidup masih ada pada koloni yang sama | 🎯 **WHITE SYNDROME** — pindahkan ke folder `ws_candidates/` |
| Kerangka mati penuh overgrowth alga/lumat | Dead lama biasa — bukan WS |
| Karang putih merata seluruh koloni | Itu bleaching salah label — catat sebagai noise |

**Aktivitas manual:** telusuri ±1.500 gambar kelas Dead/REVIEW sekali jalan (±2–3 jam kerja). Target menemukan 50–300 citra bergaya WS.

> 💡 Jika hasil kurang, tambah/perkuat dari sumber publik spesifik penyakit karang (mis. EILAT Red Sea coral disease dataset) — pendampingan pembimbingmu di BRIN sangat mungkin punya arsip foto lapangan penyakit.

---

## BAGIAN 6 — TAHAP A: Baseline Klasifikasi (bukti awal + transfer learning)

```python
# train_cls_a.py
"""Baseline klasifikasi YOLOv8-cls: Healthy vs Bleached vs Dead."""
from ultralytics import YOLO

model = YOLO("yolov8s-cls.pt")
model.train(
    data="data_tahap_a",     # folder berisi subfolder per kelas
    epochs=50,
    imgsz=224,
    device="mps",
    batch=32,
    workers=2,
    hsv_h=0.0,               # proteksi warna juga di sini
    project="runs", name="cls_baseline_A",
)
```

Hasil ini memberimu:
1. Angka baseline untuk laporan (*"klasifikasi image-level mencapai acc X%"*)
2. Cerita transfer learning menuju Tahap B
3. Bukti bahwa kelas Dead bisa dipisahkan → mendukung hipotesis WS tersembunyi di sana

---

## BAGIAN 7 — PROYEKSI KE TAHAP B: Anotasi Polygon

### Protokol inti (tetap berlaku dari v1):

| Kelas | Cara anotasi polygon |
|-------|----------------------|
| healthy | kontur koloni ber-jaringan utuh |
| bleaching | seluruh area putih MERATA, tissue intact |
| white_syndrome | hanya zona lesi: garis batas tajam jaringan hidup ↔ kerangka telanjang |

### Workflow percepatan:
1. Jalankan `label-studio start` (terminal terpisah di VS Code)
2. Template **Polygon**, 3 label (healthy hijau / bleaching kuning / white_syndrome merah)
3. Pasang SAM-assist via **AnyLabeling** untuk auto-trace koloni → kamu hanya koreksi & beri label (3–5× lebih cepat)
4. Kurasi subset: mulai dari `ws_candidates/` + sampel seimbang healthy & bleaching
5. Export JSON → simpan ke `label_studio_export/`

---

## BAGIAN 8 — KONVERSI & SPLIT

```bash
# Konversi JSON Label Studio -> .txt YOLO (polygon mode seg)
python convert_labels.py --json label_studio_export/export.json \
                         --img-dir label_studio_export/images \
                         --out-labels data/labels/all --mode seg

# Copy pasangannya lalu split stratified
cp data/labels/all/*.txt data/labels/all_tmp/  # sesuaikan struktur
python split_dataset.py    # 70/20/10, seed=42
```

`data/coral.yaml` (sesuaikan absolute path hasil `pwd`):

```yaml
path: /Users/mac/Desktop/BRIN/coral-detection/data
train: images/train
val: images/val
test: images/test
names:
  0: healthy
  1: bleaching
  2: white_syndrome
```

---

## BAGIAN 9 — TRAINING YOLOv8-seg (MPS)

```bash
# Sanity check dulu (±5–10 mnt)
yolo train model=yolov8n-seg.pt data=data/coral.yaml epochs=5 imgsz=320 batch=4 device=mps workers=2
```

```python
# train.py — FINAL
from ultralytics import YOLO
model = YOLO("yolov8m-seg.pt")
model.train(
    data="data/coral.yaml",
    epochs=200, imgsz=1024,        # lesi garis tipis butuh resolusi tinggi
    device="mps", batch=4, workers=2,
    patience=30, cos_lr=True, lr0=0.001,
    hsv_h=0.0,                     # ⭐ KRUSIAL: jangan acak hue!
    hsv_s=0.3, hsv_v=0.4,
    degrees=10, translate=0.1, scale=0.5,
    flipud=0.5, fliplr=0.5, mosaic=1.0, mixup=0.15,
    erasing=0.0,
    project="runs", name="tahap_b_final",
)
```

> Jalankan semalam. Alternatif lambat → Colab/Kaggle (`device=0`, `batch=16`, kode sama).

---

## BAGIAN 10 — EVALUASI & PREDIKSI

`val.py` dan `predict.py` tetap seperti versi sebelumnya (evaluasi `split="test"`, cetak per-class AP50/AP50-95).

**Bukti novelty yang dilaporkan:**
- Confusion matrix → tingkat kebingungan bleaching ↔ white_syndrome
- Per-class AP50 white_syndrome
- Ablation study: `hsv_h=0` vs default; polygon vs bbox

---

## BAGIAN 11 — TIMELINE (revisi)

| Minggu | Kegiatan | Output |
|--------|----------|--------|
| 1 | ✅ Setup env · dedup audit · harmonisasi v2 | manifest + dedup report |
| 2 | Verifikasi visual (hunt WS) + diskusi protokol dgn pembimbing | folder `ws_candidates/` + protokol disetujui |
| 3 | Training baseline Tahap A | angka accuracy klasifikasi |
| 4–7 | Anotasi polygon subset ±500–1000 gambar (SAM-assist) | export JSON Label Studio |
| 8 | Konversi + split + sanity check | dataset YOLO siap |
| 9–10 | Training final YOLOv8-seg | `best.pt` |
| 11 | Evaluasi + ablation + notebook analisis | confusion matrix + tabel AP |
| 12 | Penulisan laporan + demo prediksi | ✅ |

---

## ✅ CHECKLIST EKSEKUSI (urutan praktik)

| # | Langkah | Status |
|---|---------|--------|
| 1 | Setup venv + verifikasi MPS | ☐ |
| 2 | Jalankan `cek_duplikat.py` pada `dataset_coral` | ☐ |
| 3 | Jalankan `harmonize_v2.py` → `data_tahap_a/` | ☐ |
| 4 | Cek visual kelas Dead + REVIEW → kumpulkan WS candidates | ☐ |
| 5 | Setujui protokol anotasi dengan pembimbing | ☐ |
| 6 | Train baseline klasifikasi (Tahap A) | ☐ |
| 7 | Anotasi polygon Label Studio + SAM (Tahap B) | ☐ |
| 8 | Convert → split stratified → sanity check | ☐ |
| 9 | Training final semalam | ☐ |
| 10 | Evaluasi per-class + confusion matrix | ☐ |
| 11 | Prediksi demo + tulis laporan | ☐ |

---

## 📝 Perubahan utama dari v1 → v2 (siap ceritakan ke pembimbing):

1. Dataset diklarifikasi: semua klasifikasi image-level, tanpa WS eksplisit
2. Tambahan tahap audit deduplikasi (folder flat ternyata berguna!)
3. Tambahan **Tahap A baseline klasifikasi** agar ribuan gambar tetap terpakai
4. Strategi baru **hunting WS** dari kelas Dead/unhealthy + cadangan sumber publik penyakit karang
5. dataset5 & gambar tak teridentifikasi secara resmi dieksklusi demi validitas metodologi
