import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from inference import (CLASSES, MODEL_PATH, REVIEW_CONFIDENCE, REVIEW_MARGIN, ArrhythmiaPredictor,
                       ECGInputError, label_from_filename, needs_review, prepare_input)
from performance import CONFUSION_MATRIX, FIGURES, metrics_from_confusion

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", ROOT_DIR / "test_samples_csv"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT_DIR / "results"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="ECG Arrhythmia Detection API", version="1.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()],
    # Vercel production/preview domains and local development are always allowed
    allow_origin_regex=os.getenv(
        "CORS_ORIGIN_REGEX", r"https://[a-z0-9-]+\.vercel\.app|http://(localhost|127\.0\.0\.1)(:\d+)?"
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=2048)

predictor = ArrhythmiaPredictor()


def build_response(prepared, filename: str):
    beats, labels = prepared
    start = time.perf_counter()
    probs, attention = predictor.predict(beats)
    elapsed_ms = (time.perf_counter() - start) * 1000
    preds = probs.argmax(1)
    sorted_probs = np.sort(probs, axis=1)
    attention = attention / attention.max(1, keepdims=True)

    results = [
        {
            "index": i,
            "prediction": int(preds[i]),
            "confidence": float(probs[i, preds[i]]),
            "margin": float(sorted_probs[i, -1] - sorted_probs[i, -2]),
            "needs_review": needs_review(probs[i]),
            "probabilities": [round(float(p), 5) for p in probs[i]],
            "attention": np.round(attention[i], 3).tolist(),
            "true_label": labels[i],
            "values": np.round(beats[i], 4).tolist(),
        }
        for i in range(len(beats))
    ]
    counts = Counter(int(p) for p in preds)
    labelled = [r for r in results if r["true_label"] is not None]
    abnormal = len(beats) - counts.get(0, 0)

    confusion = None
    if labelled:
        confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        for r in labelled:
            confusion[r["true_label"], r["prediction"]] += 1
        confusion = confusion.tolist()

    # Everything is already plain Python types; JSONResponse skips FastAPI's slow
    # re-encoding of the large payload (noticeable on low-CPU hosting plans)
    return JSONResponse({
        "filename": filename,
        "num_beats": len(beats),
        "inference_ms": round(elapsed_ms, 1),
        "summary": {
            "counts": [counts.get(c["id"], 0) for c in CLASSES],
            "dominant_class": max(counts, key=counts.get),
            "abnormal_beats": abnormal,
            "abnormal_ratio": abnormal / len(beats),
            "mean_confidence": float(probs.max(1).mean()),
            "review_beats": sum(r["needs_review"] for r in results),
            "accuracy": (sum(r["prediction"] == r["true_label"] for r in labelled) / len(labelled))
            if labelled else None,
            "labelled_beats": len(labelled),
            "confusion_matrix": confusion,
        },
        "results": results,
    })


def run_prepare(fn, *args):
    try:
        return fn(*args)
    except ECGInputError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/info")
def info():
    return {
        "model": "ECG_CNN_LSTM_Deep (CNN + BiLSTM + Attention)",
        "parameters": predictor.num_params,
        "input_length": 187,
        "sampling_rate_hz": 125,
        "dataset": "MIT-BIH Arrhythmia (Kaggle heartbeat)",
        "weights": MODEL_PATH.name,
        "classes": CLASSES,
        "review_thresholds": {"confidence": REVIEW_CONFIDENCE, "margin": REVIEW_MARGIN},
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 20 MB.")
    name = Path(file.filename or "upload.csv").name
    return build_response(run_prepare(prepare_input, content, name), name)


# ---------------------------------------------------------------- bundled test samples

def contained_file(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise HTTPException(404, "File not found.")
    return path


@app.get("/api/samples")
def list_samples():
    if not SAMPLES_DIR.is_dir():
        return []
    return [
        {"name": p.name, "label": label_from_filename(p.name)}
        for p in sorted(SAMPLES_DIR.glob("*.csv"))
    ]


@app.get("/api/samples/{name}/download")
def download_sample(name: str):
    return FileResponse(contained_file(SAMPLES_DIR, name), media_type="text/csv", filename=name)


@app.post("/api/samples/{name}/predict")
def predict_sample(name: str):
    path = contained_file(SAMPLES_DIR, name)
    return build_response(run_prepare(prepare_input, path.read_bytes(), path.name), path.name)


# ---------------------------------------------------------------- model performance

@app.get("/api/performance")
def performance():
    return {
        "model": "ECG_CNN_LSTM_Deep",
        "dataset": "MIT-BIH heartbeat test split (Kaggle shayanfazeli/heartbeat)",
        **metrics_from_confusion(CONFUSION_MATRIX),
        "figures": [f for f in FIGURES if (RESULTS_DIR / f["file"]).is_file()],
    }


@app.get("/api/figures/{name}")
def figure(name: str):
    return FileResponse(contained_file(RESULTS_DIR, name), media_type="image/png")
