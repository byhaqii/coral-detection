# build_dataset.py
"""
Bangun data_cls/ (klasifikasi Tahap A) dari folder flat dataset_coral.
- Label dari prefix nama file
- dataset5 dieksklusi (augmented dari dataset1 -> leakage)
- Split stratified 70/20/10, seed=42
"""
import random, shutil
from collections import defaultdict
from pathlib import Path
import pandas as pd
from config import RAW_DATA, DATA_CLS

random.seed(42)
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

def parse_label(fname: str):
    n = fname.lower()
    if n.startswith("dataset5_"):
        return None
    if n.startswith("dataset2_"):
        return None          # label CORAL generik tak terverifikasi
    if "_coral_bl_" in n:      return "bleached"
    if "_bleached" in n:       return "bleached"
    if "_unhealthy" in n:      return "bleached"   # ds7: deskripsi = bleaching
    if "_dead" in n:           return "dead"
    if "_healthy" in n:        return "healthy"
    if "_unbleached" in n:     return "healthy"
    if "_coral_" in n:         return "healthy"    # ds2 CORAL polos
    return None

def main():
    per_class = defaultdict(list)
    manifest = []

    for f in sorted(RAW_DATA.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = parse_label(f.name)
        manifest.append({"filename": f.name, "label": label or "EXCLUDED"})
        if label:
            per_class[label].append(f)

    for split in SPLITS:
        for label in per_class:
            (DATA_CLS / split / label).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: defaultdict(int))
    for label, files in per_class.items():
        files = files[:]
        random.shuffle(files)
        n = len(files)
        n_train = round(n * SPLITS["train"])
        n_val = round(n * SPLITS["val"])
        assign = (["train"] * n_train + ["val"] * n_val +
                  ["test"] * (n - n_train - n_val))
        for f, split in zip(files, assign):
            shutil.copy2(f, DATA_CLS / split / label / f.name)
            stats[split][label] += 1

    pd.DataFrame(manifest).to_csv("manifest_dataset_cls.csv", index=False)

    print("\n=== DISTRIBUSI FINAL ===")
    labels = ["healthy", "bleached", "dead"]
    print(f"{'split':7s}" + "".join(f"{l:>12s}" for l in labels))
    for split in SPLITS:
        print(f"{split:7s}" + "".join(f"{stats[split][l]:12d}" for l in labels))

if __name__ == "__main__":
    main()
