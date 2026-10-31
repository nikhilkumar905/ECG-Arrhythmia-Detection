"""Loading the trained model and turning uploaded heartbeats into predictions.

Input is one heartbeat per row with 187 values (MIT-BIH format, 125 Hz, normalised
to [0, 1], zero-padded). A 188th column is treated as the true label.
"""
import io
import os
import re
from pathlib import Path

import joblib
import numpy as np
import torch

from model_definition import ECG_CNN_LSTM_Deep

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR.parent / "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "best_model.pth"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", MODEL_DIR / "scaler.pkl"))

BEAT_LEN = 187
MAX_BEATS = 5000
INFER_BATCH = int(os.getenv("INFER_BATCH", "64"))

# Shared/fractional-CPU hosts (Render sets RENDER=true) run faster and leaner on one thread
if os.getenv("TORCH_THREADS") or os.getenv("RENDER"):
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", "1")))

# A prediction is flagged for review when the model is unsure or torn between two classes
REVIEW_CONFIDENCE = 0.70
REVIEW_MARGIN = 0.20

CLASSES = [
    {
        "id": 0, "code": "N", "name": "Normal",
        "description": "Normal sinus beat, including bundle branch block and nodal escape beats.",
        "severity": "normal",
    },
    {
        "id": 1, "code": "S", "name": "Supraventricular",
        "description": "Supraventricular ectopic beat - premature beat originating above the ventricles (atria or AV node).",
        "severity": "moderate",
    },
    {
        "id": 2, "code": "V", "name": "Ventricular",
        "description": "Ventricular ectopic beat (PVC) - premature beat originating in the ventricles.",
        "severity": "high",
    },
    {
        "id": 3, "code": "F", "name": "Fusion",
        "description": "Fusion of a ventricular and a normal beat occurring at the same time.",
        "severity": "moderate",
    },
    {
        "id": 4, "code": "Q", "name": "Unknown",
        "description": "Unclassifiable beat, including paced beats.",
        "severity": "unknown",
    },
]

class ECGInputError(ValueError):
    pass


class ArrhythmiaPredictor:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.device = torch.device("cpu")
        self.model = ECG_CNN_LSTM_Deep(num_classes=5, dropout=0.45)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.num_params = sum(p.numel() for p in self.model.parameters())

        # Capture the attention scores of each forward pass without changing the model
        self._attention_scores = None
        self.model.attention.register_forward_hook(
            lambda _m, _inp, out: setattr(self, "_attention_scores", out)
        )

    @torch.no_grad()
    def predict(self, beats):
        """Return (probabilities (N, 5), attention weights (N, T)) for 187-sample beats.

        The attention weights cover the T=23 LSTM time steps; each step corresponds to
        ~8 input samples after the three pooling layers.
        """
        x = self.scaler.transform(beats.astype(np.float64))
        x = torch.from_numpy(x).float().unsqueeze(1)  # (N, 1, 187)
        probs, attention = [], []
        # Small batches keep peak memory low enough for 512 MB hosting plans
        for i in range(0, len(x), INFER_BATCH):
            probs.append(torch.softmax(self.model(x[i:i + INFER_BATCH]), dim=1))
            attention.append(torch.softmax(self._attention_scores, dim=1).squeeze(-1))
        return torch.cat(probs).numpy(), torch.cat(attention).numpy()


def needs_review(probs_row):
    top2 = np.sort(probs_row)[-2:]
    return bool(top2[1] < REVIEW_CONFIDENCE or top2[1] - top2[0] < REVIEW_MARGIN)


# ---------------------------------------------------------------- file parsing

def parse_numeric(content: bytes, filename: str) -> np.ndarray:
    """Return a 2-D float array from a CSV / TXT / NPY upload, skipping header lines."""
    if filename.lower().endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(content), allow_pickle=False)
        except Exception as exc:
            raise ECGInputError(f"Could not read .npy file: {exc}")
        arr = np.asarray(arr, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ECGInputError("File is not valid text. Upload a CSV, TXT or NPY file.")

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,;\t ]+", line) if p]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            pass  # header or comment line
    if not rows:
        raise ECGInputError("No numeric values found in the file.")

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        width = max(widths, key=lambda w: sum(len(r) == w for r in rows))
        rows = [r for r in rows if len(r) == width]
    return np.asarray(rows, dtype=np.float64)


def label_from_filename(filename: str):
    m = re.search(r"label[_-]?(\d)", filename, re.IGNORECASE)
    return int(m.group(1)) if m and int(m.group(1)) < len(CLASSES) else None


def prepare_input(content: bytes, filename: str):
    """Return (beats, true_labels) for a CSV / TXT / NPY file of heartbeats."""
    data = parse_numeric(content, filename)
    n_rows, n_cols = data.shape
    if n_cols not in (BEAT_LEN, BEAT_LEN + 1):
        raise ECGInputError(
            f"Each row must be one heartbeat with {BEAT_LEN} values (optionally followed by a label "
            f"0-4), but the file has {n_cols} value{'' if n_cols == 1 else 's'} per row."
        )

    beats = data[:, :BEAT_LEN]
    labels = [None] * n_rows
    if n_cols == BEAT_LEN + 1:
        col = data[:, BEAT_LEN]
        if np.all((col == np.round(col)) & (col >= 0) & (col < len(CLASSES))):
            labels = [int(v) for v in col]
    elif n_rows == 1:
        labels = [label_from_filename(filename)]
    if n_rows > MAX_BEATS:
        raise ECGInputError(f"The file has {n_rows:,} beats; the limit is {MAX_BEATS:,} per upload.")
    return beats, labels
