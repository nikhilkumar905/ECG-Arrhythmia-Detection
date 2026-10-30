"""Loading the trained model and turning uploaded files into predictions.

Three kinds of input are accepted:
  * Heartbeat segments - one beat per row with 187 values (MIT-BIH format, 125 Hz,
    normalised to [0, 1], zero-padded). A 188th column is treated as the true label.
  * A continuous ECG recording - a single column or a single long row.
  * A PhysioNet WFDB record (.hea + .dat, optionally .atr beat annotations).
Continuous signals are resampled to 125 Hz, R-peaks are detected (or taken from the
annotations) and each beat is cut out the same way the MIT-BIH heartbeat dataset was
built (Kachuee et al., 2018).
"""
import io
import os
import re
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

from model_definition import ECG_CNN_LSTM_Deep

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR.parent / "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "best_model.pth"))
SCALER_PATH = Path(os.getenv("SCALER_PATH", MODEL_DIR / "scaler.pkl"))

BEAT_LEN = 187
TARGET_FS = 125
MAX_BEATS = 5000
MAX_SIGNAL_SEC = 60 * 60

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

# MIT-BIH annotation symbols -> AAMI class (as used to build the Kaggle heartbeat dataset)
AAMI_MAP = {
    **dict.fromkeys("NLRej", 0),
    **dict.fromkeys("AaJS", 1),
    **dict.fromkeys("VE", 2),
    "F": 3,
    **dict.fromkeys("/fQ", 4),
}


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
        for i in range(0, len(x), 1024):
            probs.append(torch.softmax(self.model(x[i:i + 1024]), dim=1))
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


def bandpass(signal, fs):
    nyq = fs / 2
    high = min(40.0, nyq * 0.9)
    b, a = butter(2, [0.5 / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal) if len(signal) > 15 else signal


def segment_signal(signal: np.ndarray, fs: float, ann_samples=None):
    """Cut a continuous ECG into MIT-BIH style 187-sample beats starting at each R-peak.

    `ann_samples` (R-peak positions in the original sampling rate) replaces peak
    detection when the recording comes with beat annotations.
    Returns (beats, beat_lengths, normalised 125 Hz signal, peak indices, kept annotation idx).
    """
    if len(signal) < fs * 2:
        raise ECGInputError("Signal is too short - provide at least 2 seconds of ECG.")
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float64))
    signal = signal[:int(MAX_SIGNAL_SEC * fs)]

    filtered = bandpass(signal - np.median(signal), fs)
    if fs != TARGET_FS:
        up, down = TARGET_FS, int(round(fs))
        g = np.gcd(up, down)
        sig = resample_poly(filtered, up // g, down // g)
    else:
        sig = filtered

    # Flip inverted leads so R-peaks point upwards
    if abs(np.percentile(sig, 0.5)) > abs(np.percentile(sig, 99.5)):
        sig = -sig

    # Normalise to [0, 1] in 10 s windows, as in the original dataset
    norm = np.zeros_like(sig)
    win = TARGET_FS * 10
    for start in range(0, len(sig), win):
        w = sig[start:start + win]
        lo, hi = np.percentile(w, 0.5), np.max(w)
        norm[start:start + win] = np.clip((w - lo) / (hi - lo + 1e-8), 0, 1)

    if ann_samples is not None:
        peaks = np.round(np.asarray(ann_samples) * TARGET_FS / fs).astype(int)
        keep = np.where((peaks >= 0) & (peaks < len(norm) - 1))[0]
        peaks = peaks[keep]
    else:
        peaks, _ = find_peaks(norm, height=0.5, distance=int(0.25 * TARGET_FS), prominence=0.3)
        keep = np.arange(len(peaks))
    if len(peaks) < 2:
        raise ECGInputError("Could not detect heartbeats (R-peaks) in the signal. "
                            "Check the sampling rate or upload pre-segmented beats.")
    peaks, keep = peaks[:MAX_BEATS], keep[:MAX_BEATS]

    # Beat length = 1.2 x median RR interval of the 10 s window the beat falls in
    rr = np.diff(peaks)
    window_of = peaks[:-1] // win
    global_rr = np.median(rr)
    beats, lengths = [], []
    for p in peaks:
        local = rr[window_of == p // win]
        median_rr = np.median(local) if len(local) else global_rr
        length = int(min(BEAT_LEN, round(1.2 * median_rr)))
        seg = norm[p:p + length]
        beat = np.zeros(BEAT_LEN)
        beat[:len(seg)] = seg
        beats.append(beat)
        lengths.append(len(seg))
    return np.asarray(beats), lengths, norm, peaks, keep


def label_from_filename(filename: str):
    m = re.search(r"label[_-]?(\d)", filename, re.IGNORECASE)
    return int(m.group(1)) if m and int(m.group(1)) < len(CLASSES) else None


def signal_info(norm, peaks, fs, lengths, extra=None):
    return {
        "fs_original": fs,
        "fs_resampled": TARGET_FS,
        "duration_sec": round(len(norm) / TARGET_FS, 2),
        "signal": np.round(norm, 3).tolist(),
        "peaks": peaks.tolist(),
        "beat_lengths": lengths,
        "heart_rate_bpm": round(60 * TARGET_FS / float(np.median(np.diff(peaks))), 1),
        **(extra or {}),
    }


def prepare_input(content: bytes, filename: str, fs: float):
    """Return (beats, true_labels, mode, signal_info) for a single CSV / TXT / NPY file."""
    data = parse_numeric(content, filename)
    n_rows, n_cols = data.shape

    if n_cols in (BEAT_LEN, BEAT_LEN + 1):
        beats = data[:, :BEAT_LEN]
        labels = [None] * n_rows
        if n_cols == BEAT_LEN + 1:
            col = data[:, BEAT_LEN]
            if np.all((col == np.round(col)) & (col >= 0) & (col < len(CLASSES))):
                labels = [int(v) for v in col]
        elif n_rows == 1:
            labels = [label_from_filename(filename)]
        if n_rows > MAX_BEATS:
            beats, labels = beats[:MAX_BEATS], labels[:MAX_BEATS]
        return beats, labels, "beats", None

    if n_cols == 1 or n_rows == 1:
        signal = data.ravel()
    elif n_cols <= 13 and n_rows > n_cols:
        # Multi-column recording (e.g. time + leads): use the column with the most variance,
        # skipping a monotonic time column
        candidates = [c for c in range(n_cols) if not np.all(np.diff(data[:, c]) > 0)] or list(range(n_cols))
        signal = data[:, max(candidates, key=lambda c: np.std(data[:, c]))]
    else:
        raise ECGInputError(
            f"Unrecognised shape {n_rows}x{n_cols}. Upload beats with 187 values per row "
            "(optionally + label) or a single-column continuous ECG signal."
        )

    beats, lengths, norm, peaks, _ = segment_signal(signal, fs)
    return beats, [None] * len(beats), "signal", signal_info(norm, peaks, fs, lengths)


def prepare_wfdb(files: dict):
    """Return (beats, true_labels, mode, signal_info) for a WFDB record given {filename: bytes}."""
    import wfdb

    headers = [n for n in files if n.lower().endswith(".hea")]
    if len(headers) != 1:
        raise ECGInputError("Upload exactly one WFDB record: its .hea and .dat files (and optionally .atr).")
    record_name = Path(headers[0]).stem

    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            (Path(tmp) / Path(name).name).write_bytes(content)
        base = str(Path(tmp) / record_name)
        try:
            record = wfdb.rdrecord(base)
        except FileNotFoundError:
            raise ECGInputError(f"The signal file for record '{record_name}' is missing - upload the .dat file too.")
        except Exception as exc:
            raise ECGInputError(f"Could not read WFDB record '{record_name}': {exc}")
        ann = None
        if (Path(tmp) / f"{record_name}.atr").exists():
            try:
                ann = wfdb.rdann(base, "atr")
            except Exception as exc:
                raise ECGInputError(f"Could not read annotations '{record_name}.atr': {exc}")

    leads = [s.upper() for s in record.sig_name]
    lead_idx = leads.index("MLII") if "MLII" in leads else 0
    signal = record.p_signal[:, lead_idx]
    fs = float(record.fs)

    ann_samples, labels, symbols = None, None, None
    if ann is not None:
        beat_mask = [s in AAMI_MAP for s in ann.symbol]
        ann_samples = np.asarray(ann.sample)[beat_mask]
        symbols = [s for s, m in zip(ann.symbol, beat_mask) if m]

    beats, lengths, norm, peaks, keep = segment_signal(signal, fs, ann_samples)
    if symbols is not None:
        symbols = [symbols[i] for i in keep]
        labels = [AAMI_MAP[s] for s in symbols]
    else:
        labels = [None] * len(beats)

    info = signal_info(norm, peaks, fs, lengths, {
        "record": record_name,
        "lead": record.sig_name[lead_idx],
        "annotated": ann is not None,
        "symbols": symbols,
    })
    return beats, labels, "record", info
