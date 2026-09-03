# train_cls_a.py
"""Baseline klasifikasi Tahap A: Healthy vs Bleached vs Dead."""
from ultralytics import YOLO
from config import DATA_CLS

model = YOLO("yolov8s-cls.pt")
model.train(
    data=str(DATA_CLS),
    epochs=30,
    imgsz=224,
    device="mps",
    batch=32,
    workers=2,
    hsv_h=0.0,            # ⭐ proteksi warna (fitur pembeda utama)
    project="runs", name="cls_baseline_A",
)

# Evaluasi di test set (data yang belum pernah dilihat model)
best = YOLO("runs/cls_baseline_A/weights/best.pt")
metrics = best.val(data=str(DATA_CLS), split="test", device="mps")
print("\n=== HASIL BASELINE ===")
print(f"Top-1 accuracy : {metrics.top1:.4f}")
print(f"Top-5 accuracy : {metrics.top5:.4f}")
