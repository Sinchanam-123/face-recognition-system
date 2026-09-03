# Legacy notebooks (archived — do not run)

These are the **original prototypes** of the attendance system, built on
`face_recognition` / **dlib**. They are kept for reference and history only.
They have been **superseded by `backend/`**, which is the system that actually
runs.

**They are not runnable on current Python.** dlib has no prebuilt wheel for
Python 3.13+, which is what forced the rewrite onto InsightFace (ArcFace on
onnxruntime). `encode_faces.ipynb` even pins a hand-built dlib wheel for
cp313 from a third-party GitHub release — that URL is the only reason the
notebook ever installed at all.

| Notebook | What it did | Replaced by |
|---|---|---|
| `download_dataset.ipynb` | Fetched the sklearn LFW dataset and wrote it to `known_faces/` | — (dataset step no longer used) |
| `encode_faces.ipynb` | Built `encodings.pkl` — 128-d dlib face encodings | `backend/encode_faces.py` (512-d ArcFace → `face_db.pkl`) |
| `attendance_system.ipynb` | Live recognition loop with a blocking `input()` prompt to name unknown faces | `backend/engine.py` + `backend/app.py` (camera thread + REST API) + the React UI's unknown-face cards |

## Known bug: `download_dataset.ipynb` produced 3023 all-black images

This is why `known_faces/` is worthless and is no longer tracked in the repo.

The notebook writes the LFW images out like this:

```python
lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)
...
img_bgr = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2BGR)
cv2.imwrite(f"{person_dir}/{idx}.jpg", img_bgr)
```

`fetch_lfw_people` returns pixel data as **floats in the range 0.0–1.0**.
Calling `.astype('uint8')` on that **truncates toward zero**, so every pixel
below 1.0 becomes `0` — i.e. every pixel. The result is 3023 completely black
125×94 JPEGs across 62 people (verified: mean ≈ 0.00, std ≈ 0.00).

The fix would have been to rescale before casting:

```python
img_uint8 = (image * 255).astype('uint8')   # correct
```

Note that `encode_faces.ipynb` cell 5 *does* apply the `* 255` fix and encodes
straight from the in-memory LFW arrays — which is why `encodings.pkl` contains
real encodings even though the `known_faces/` folder on disk is blank. Those
128-d dlib vectors are not compatible with ArcFace's 512-d embeddings, so
`encodings.pkl` is dead too.

To enrol faces now, either register them live through the web UI, or point
`backend/encode_faces.py` at a folder of real photos.
