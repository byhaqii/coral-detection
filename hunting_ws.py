# hunting_ws.py — FINAL, copy seluruh ini
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

KELAS  = "dead"       # ganti: "bleached"
SPLIT  = "train"      # lalu "val", "test"
FOLDER = Path(f"data_cls/{SPLIT}/{KELAS}")
FILES  = sorted(FOLDER.glob("*"))
BATCH  = 10

kandidat = Path("ws_candidates.txt")
mulai = 0

while mulai < len(FILES):
    batch = FILES[mulai:mulai+BATCH]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for nomor, (ax, f) in enumerate(zip(axes.flat, batch)):
        ax.imshow(Image.open(f))
        ax.set_title(f"[{nomor}] {f.name[:26]}", fontsize=7)
        ax.text(
            0.03, 0.95, str(nomor), transform=ax.transAxes,
            ha="left", va="top", fontsize=16, fontweight="bold",
            color="black", bbox={"facecolor": "white", "alpha": 0.85, "pad": 3},
        )
        ax.axis("off")
    plt.suptitle(f"{SPLIT}/{KELAS} — batch mulai indeks {mulai} dari {len(FILES)}")
    plt.show(block=False)
    plt.pause(1.0)

    jawab = input(f"\nNomor kandidat WS (0-{len(batch)-1}, spasi) | Enter=lanjut | b=back | q=quit: ").strip()

    if jawab.lower() == "q":
        break
    if jawab.lower() == "b":
        mulai = max(0, mulai - BATCH)
        plt.close("all")
        continue

    for n in jawab.split():
        f = batch[int(n)]
        with open(kandidat, "a") as fh:
            fh.write(f"{SPLIT}/{KELAS}/{f.name}\n")
        print(f"  ✅ tersimpan: {f.name[:44]}")

    mulai += BATCH
    plt.close("all")

jumlah = len(kandidat.read_text().splitlines()) if kandidat.exists() else 0
print(f"\nSelesai. Total kandidat di ws_candidates.txt: {jumlah}")
