# eval_cls_a.py — evaluasi baseline di test set
from ultralytics import YOLO

best = YOLO("runs/classify/runs/cls_baseline_A/weights/best.pt")
metrics = best.val(data="data_cls", split="test", device="mps")
print("\n=== HASIL BASELINE (test set) ===")
print(f"Top-1 accuracy : {metrics.top1:.4f}")
print(f"Top-5 accuracy : {metrics.top5:.4f}")
