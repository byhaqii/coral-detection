# eval_cls_a_v2.py — evaluasi baseline v2 di test set
from ultralytics import YOLO

best = YOLO("runs/classify/runs/cls_baseline_A-2/weights/best.pt")
metrics = best.val(data="data_cls", split="test", device="mps")
print("\n=== HASIL BASELINE v2 (test set) ===")
print(f"Top-1 accuracy : {metrics.top1:.4f}")
