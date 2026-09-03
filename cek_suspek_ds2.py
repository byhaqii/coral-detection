# cek_suspek_ds2.py — visualkan kesalahan yang berasal dari dataset2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

suspects = [
    "dataset2_train_CORAL_LIS4008A_2014_11888",
    "dataset2_test_CORAL_MAI-B2488_2019_11_16",
    "dataset2_train_CORAL_MAI-B2488_2019_06_1",
    "dataset2_test_CORAL_MAI-B4025_2019_06_25",
    "dataset2_test_CORAL_OAH-B3078_2019_18_34",
    "dataset2_train_CORAL_KUR-B010_2019_18_53",
    "dataset1_healthy_corals_38662947541_c1f9",   # pembanding: satu dari ds1
]
for name in suspects:
    matches = list(Path("data_cls").rglob(f"*{name}*"))
    if not matches:
        print(f"[tidak ketemu] {name}")
        continue
    f = matches[0]
    plt.figure(figsize=(4, 3))
    plt.imshow(Image.open(f))
    plt.title(f"{f.parent.name} | {f.name[:40]}", fontsize=8)
    plt.axis("off")
    plt.show()
    print(f"{name} -> periksa: putih/bleached atau sehat?")
