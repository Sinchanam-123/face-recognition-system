"""
Build the face database (face_db.pkl) from a folder of real photos.

This is the InsightFace/3.14 replacement for `encode_faces.ipynb`. It scans a
dataset laid out as one sub-folder per person:

    dataset/
      Alice/  img1.jpg  img2.jpg ...
      Bob/    img1.png  ...

and writes 512-d ArcFace embeddings to face_db.pkl (the same file the running
app reads and appends to when you register faces live).

Usage:
    py backend/encode_faces.py                 # uses ./dataset by default
    py backend/encode_faces.py path/to/dataset

Note: images must be reasonably sized, real face photos. The original
`known_faces/` folder in this repo contains empty/black files and will yield
nothing — drop proper photos in a new folder instead.
"""

import os
import pickle
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "face_db.pkl")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(dataset_path):
    if not os.path.isdir(dataset_path):
        print(f"❌ Dataset folder not found: {dataset_path}")
        sys.exit(1)

    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    embeddings, names = [], []
    total, skipped = 0, 0

    for person in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        count = 0
        for fname in os.listdir(person_dir):
            if os.path.splitext(fname)[1].lower() not in EXTS:
                continue
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                skipped += 1
                continue
            faces = app.get(img)
            if faces:
                # Keep the largest / most confident face in the photo.
                face = max(faces, key=lambda f: float(f.det_score))
                embeddings.append(face.normed_embedding)
                names.append(person)
                total += 1
                count += 1
            else:
                skipped += 1
        print(f"✅ {person}: {count} encoded")

    with open(DB_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "names": names}, f)

    print(f"\n🎉 Done! {total} encodings saved to {DB_PATH}")
    print(f"⚠️  Skipped {skipped} images (no detectable face).")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "dataset")
    main(dataset)
