# per_class_eval.py — accuracy PER KELAS (angka yang jujur)
from ultralytics import YOLO
from pathlib import Path
import torch

model = YOLO("runs/classify/runs/cls_baseline_A-2/weights/best.pt")
names = model.names  # {0: 'bleached', 1: 'dead', 2: 'healthy', ...}

correct, total = {}, {}
for f in Path("data_cls/test").rglob("*"):
    if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    true_label = f.parent.name
    r = model(str(f), verbose=False)
    pred_label = names[r[0].probs.top1]
    total[true_label] = total.get(true_label, 0) + 1
    if pred_label == true_label:
        correct[true_label] = correct.get(true_label, 0) + 1

print(f"{'kelas':10s} {'benar':>7s} {'total':>7s} {'acc':>8s}")
for k in sorted(total):
    acc = correct.get(k, 0) / total[k]
    print(f"{k:10s} {correct.get(k,0):7d} {total[k]:7d} {acc:8.1%}")
