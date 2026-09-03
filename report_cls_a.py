"""Buat artefak presentasi evaluasi test set untuk baseline klasifikasi A.

Output ditulis ke reports/cls_baseline_A_test/ dan memakai model terbaik baseline A.
Baris confusion matrix = label sebenarnya, kolom = prediksi model.
"""
from pathlib import Path
import csv
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "runs/classify/runs/cls_baseline_A/weights/best.pt"
TEST_DIR = ROOT / "data_cls/test"
RUN_DIR = ROOT / "runs/classify/runs/cls_baseline_A"
OUT_DIR = ROOT / "reports/cls_baseline_A_test"
ORDER = ["healthy", "bleached", "dead"]
DISPLAY = {"healthy": "Healthy", "bleached": "Bleached", "dead": "Dead"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    files = sorted(p for p in TEST_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    actual = [p.parent.name for p in files]
    predicted, confidence = [], []

    # Batch inference keeps the report reproducible without modifying the dataset.
    for start in range(0, len(files), 32):
        batch = files[start : start + 32]
        results = model([str(p) for p in batch], verbose=False, batch=32)
        for result in results:
            top1 = int(result.probs.top1)
            predicted.append(model.names[top1])
            confidence.append(float(result.probs.top1conf))

    index = {name: i for i, name in enumerate(ORDER)}
    cm = np.zeros((3, 3), dtype=int)
    for truth, pred in zip(actual, predicted):
        cm[index[truth], index[pred]] += 1

    correct = int(np.trace(cm))
    total = len(files)
    accuracy = correct / total
    with (OUT_DIR / "confusion_matrix_test.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *[DISPLAY[name] for name in ORDER]])
        for name, row in zip(ORDER, cm):
            writer.writerow([DISPLAY[name], *row])

    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Jumlah gambar")
    ax.set(xticks=range(3), yticks=range(3),
           xticklabels=[DISPLAY[x] for x in ORDER],
           yticklabels=[DISPLAY[x] for x in ORDER],
           xlabel="Prediksi model", ylabel="Label sebenarnya",
           title="Confusion Matrix — Test Set (Baseline A)")
    threshold = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix_test.png", dpi=200)
    plt.close(fig)

    errors = [(p, t, pred, conf) for p, t, pred, conf in zip(files, actual, predicted, confidence) if t != pred]
    # Highest-confidence errors are useful to discuss: they reveal systematic, not merely ambiguous, mistakes.
    errors.sort(key=lambda x: x[3], reverse=True)
    chosen = errors[:12]
    if chosen:
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        for ax, item in zip(axes.flat, chosen):
            path, truth, pred, conf = item
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_title(f"True: {DISPLAY[truth]}\nPred: {DISPLAY[pred]} ({conf:.1%})", fontsize=10)
            ax.axis("off")
        for ax in axes.flat[len(chosen):]:
            ax.axis("off")
        fig.suptitle("Contoh Salah Klasifikasi — Test Set", fontsize=16)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "misclassified_examples.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    counts = {split: {name: len([p for p in (ROOT / "data_cls" / split / name).iterdir() if p.suffix.lower() in IMAGE_EXTS])
                      for name in ORDER}
              for split in ("train", "val", "test")}
    with (OUT_DIR / "summary.md").open("w") as f:
        f.write("# Baseline A — Evaluasi Test Set\n\n")
        f.write(f"Top-1 accuracy: **{accuracy:.2%}** ({correct}/{total})\n\n")
        f.write("## Jumlah data\n\n| Split | Healthy | Bleached | Dead | Total |\n|---|---:|---:|---:|---:|\n")
        for split, d in counts.items():
            f.write(f"| {split} | {d['healthy']} | {d['bleached']} | {d['dead']} | {sum(d.values())} |\n")
        f.write("\n## Confusion matrix\n\nBaris = label sebenarnya; kolom = prediksi model.\n\n")
        f.write("| Actual \\ Predicted | Healthy | Bleached | Dead |\n|---|---:|---:|---:|\n")
        for name, row in zip(ORDER, cm):
            f.write(f"| {DISPLAY[name]} | " + " | ".join(map(str, row)) + " |\n")
        f.write("\n## Per kelas\n\n| Kelas | Benar/Total | Recall |\n|---|---:|---:|\n")
        for name, row in zip(ORDER, cm):
            f.write(f"| {DISPLAY[name]} | {row[index[name]]}/{row.sum()} | {row[index[name]] / row.sum():.2%} |\n")
        f.write("\nArtefak: `confusion_matrix_test.png`, `misclassified_examples.png`, dan training curve `results.png`.\n")

    # Retain a presentation-local copy of the already generated training curve.
    shutil.copy2(RUN_DIR / "results.png", OUT_DIR / "training_curve.png")
    print(f"Top-1 test accuracy: {accuracy:.4f} ({correct}/{total})")
    print(cm)
    print(f"Errors: {len(errors)}; report: {OUT_DIR}")


if __name__ == "__main__":
    main()
