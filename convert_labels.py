"""
Konversi export JSON Label Studio -> format YOLO (bbox & segmentation).
Usage:
    python convert_labels.py --json label_studio_export/export.json \
                             --img-dir label_studio_export/images \
                             --out-labels data/labels/all \
                             --mode seg      # atau 'bbox'
"""
import json
import os
import argparse
from pathlib import Path
from PIL import Image

CLASSES = {"healthy": 0, "bleaching": 1, "white_syndrome": 2}


def get_image_size(img_dir: str, fname: str):
    img = Image.open(os.path.join(img_dir, fname))
    return img.size  # (w, h)


def process_task(task: dict, img_dir: str, out_dir: str, mode: str):
    # Ambil nama file dari URL gambar Label Studio
    raw_path = task["data"]["image"]
    fname = Path(raw_path.replace("/data/local-files/?d=", "")
                 .replace("/data/upload/", "")).name

    img_path = os.path.join(img_dir, fname)
    if not os.path.exists(img_path):
        print(f"[SKIP] Gambar tidak ditemukan: {fname}")
        return False

    w, h = get_image_size(img_dir, fname)
    lines = []

    for ann in task.get("annotations", []):
        for res in ann.get("result", []):
            value = res["value"]
            labels = value.get("polygonlabels") or value.get("rectanglelabels")
            if not labels or labels[0] not in CLASSES:
                continue
            cls_id = CLASSES[labels[0]]

            if res["type"] == "rectanglelabels":
                x, y, bw, bh = value["x"], value["y"], value["width"], value["height"]
                xc, yc = (x + bw / 2) / 100, (y + bh / 2) / 100
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw/100:.6f} {bh/100:.6f}")

            elif res["type"] == "polygonlabels":
                pts = value["points"]  # list of [x%, y%]
                if mode == "bbox":
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    xc = (x_min + x_max) / 200
                    yc = (y_min + y_max) / 200
                    lines.append(f"{cls_id} {xc:.6f} {yc:.6f} "
                                 f"{(x_max-x_min)/100:.6f} {(y_max-y_min)/100:.6f}")
                else:  # mode seg
                    coords = []
                    for px, py in pts:
                        coords.extend([px / 100 * w, py / 100 * h])
                    seg = " ".join(f"{c:.6f}" for c in coords)
                    lines.append(f"{cls_id} {seg}")

    if not lines:
        print(f"[WARN] Tidak ada anotasi valid: {fname}")
        return False

    out_file = os.path.join(out_dir, Path(fname).with_suffix(".txt"))
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out-labels", default="data/labels/all")
    parser.add_argument("--mode", choices=["seg", "bbox"], default="seg")
    args = parser.parse_args()

    os.makedirs(args.out_labels, exist_ok=True)

    with open(args.json) as f:
        tasks = json.load(f)

    ok = sum(process_task(t, args.img_dir, args.out_labels, args.mode)
             for t in tasks)
    print(f"\n✅ Berhasil konversi {ok}/{len(tasks)} gambar -> {args.out_labels}")


if __name__ == "__main__":
    main()
