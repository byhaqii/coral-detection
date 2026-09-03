# Tampilkan gambar bleached yang salah dikenali
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

model = YOLO("runs/classify/runs/cls_baseline_A/weights/best.pt")
names = model.names

errors = []
for f in Path("data_cls/test").rglob("*"):
    if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    true_label = f.parent.name
    r = model(str(f), verbose=False)
    pred = names[r[0].probs.top1]
    if pred != true_label:
        errors.append((f, true_label, pred,
                       r[0].probs.top1conf.item()))

print(f"Total kesalahan: {len(errors)}")
for f, t, p, conf in errors[:12]:
    print(f"  {t:>8s} -> {p:<8s} (conf {conf:.0%})  {f.name[:40]}")

# visualkan 12 kesalahan pertama
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for ax, (f, t, p, conf) in zip(axes.flat, errors[:12]):
    ax.imshow(Image.open(f))
    ax.set_title(f"TRUE:{t}\nPRED:{p} ({conf:.0%})", fontsize=8)
    ax.axis("off")
plt.tight_layout(); plt.show()
