# puts the dataset files into one zip so it can be uploaded (Zenodo / Google Drive)
# usage: python scripts/package_dataset.py
# also writes a MANIFEST.csv with the size and sha256 of every file

import argparse
import csv
import hashlib
import io
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# files / folders to include (only the ones that exist are added)
DATASET_PATHS = [
    "patches_v5/train_v5.npz",
    "patches_v4/val_v4.npz",
    "patches_v4/test_v4.npz",
    "data/patches/train.npz",
    "data/patches/val.npz",
    "data/patches/test.npz",
    "monthly_inputs",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def find_files():
    files = []
    for rel in DATASET_PATHS:
        p = PROJECT_DIR / rel
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    files.append(f)
    return files


parser = argparse.ArgumentParser()
parser.add_argument("--out", default=str(PROJECT_DIR / "data" / "sentinel1_lulc_dataset.zip"))
args = parser.parse_args()

files = find_files()
if len(files) == 0:
    raise SystemExit("No dataset files found. Run notebook 01 or copy the patches_* folders here.")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

manifest = io.StringIO()
writer = csv.writer(manifest)
writer.writerow(["path", "bytes", "sha256"])

# npz files are already compressed, so ZIP_STORED is enough (and much faster)
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
    for f in files:
        rel = f.relative_to(PROJECT_DIR).as_posix()
        size = f.stat().st_size
        writer.writerow([rel, size, sha256(f)])
        zf.write(f, rel)
        print(f"  + {rel}  ({size / 1e6:.1f} MB)")
    zf.writestr("MANIFEST.csv", manifest.getvalue())

print(f"\nArchive: {out}  ({out.stat().st_size / 1e9:.2f} GB)")
print("sha256: ", sha256(out))
