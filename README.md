# ECG Arrhythmia Detection

A full-stack web application that classifies ECG heartbeats into the five AAMI arrhythmia classes with a deep
**CNN + BiLSTM + Attention** network trained on the MIT-BIH Arrhythmia Database.
Upload single beats, beat batches, raw ECG recordings or PhysioNet records and get per-beat predictions,
confidence scores and model-attention visualisations.

**Repository:** https://github.com/nikhilkumar905/ECG-Arrhythmia-Detection

**Live demo:** https://ecg-arrhythmia-detection-weld.vercel.app/

![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)

![Analysis of MIT-BIH record 208](docs/analysis.png)

## Classes

| Code | Class | Description |
|---|---|---|
| N | Normal | Normal beat, including bundle branch block and nodal escape beats |
| S | Supraventricular | Premature beat originating above the ventricles |
| V | Ventricular | Premature ventricular contraction (PVC) |
| F | Fusion | Fusion of a ventricular and a normal beat |
| Q | Unknown | Unclassifiable or paced beat |

## Features

- **Flexible input**: single beats or batches (187 samples per row), raw continuous ECG (CSV / TXT / NPY) with
  automatic R-peak detection, and PhysioNet WFDB records (`.hea` + `.dat`, optional `.atr` annotations)
- **Per-beat results**: predicted class, confidence ring and probabilities for all five classes
- **Model attention**: highlights the parts of each beat the attention layer focused on
- **Needs-review flag**: marks beats with low confidence (< 70%) or two close top classes (margin < 20%)
- **Recording viewer**: 12-second ECG window, colour-coded beat markers, whole-recording overview and
  ← / → keyboard navigation
- **Accuracy vs. labels**: confusion matrix and mismatch filter whenever true labels or annotations are provided
- **Model performance page**: test-set accuracy, F1, per-class precision / recall / specificity and training figures
- **Built-in examples**: 100 test heartbeats and three 30-minute MIT-BIH recordings (100, 208, 232)
- **CSV export** of all predictions; responsive layout with dark and light themes

| Single beat | Model performance |
|---|---|
| ![Single beat prediction](docs/beat.png) | ![Model performance page](docs/performance.png) |

## Model

| | |
|---|---|
| Architecture | 4 × Conv1d blocks → 2-layer BiLSTM (192 units/direction) → attention pooling → 4 FC layers |
| Parameters | 2.65 M |
| Input | One heartbeat: 187 samples at 125 Hz, scaled 0–1, standardised per time step |
| Training | AdamW, OneCycle LR, label smoothing, EMA weights, on-the-fly augmentation, mixed precision |
| Dataset | MIT-BIH Arrhythmia Database, Kaggle pre-segmented heartbeat split (21,892 test beats) |

**Test-set results**

| Accuracy | Macro F1 | Weighted F1 |
|---|---|---|
| 99.00% | 94.0% | 99.0% |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 99.4% | 99.6% | 99.5% |
| Supraventricular | 91.8% | 86.7% | 89.2% |
| Ventricular | 97.4% | 96.5% | 97.0% |
| Fusion | 82.1% | 87.7% | 84.8% |
| Unknown | 99.8% | 99.6% | 99.7% |

The training notebook is [`ecg(2).ipynb`](ecg(2).ipynb) (Google Colab, T4 GPU).

## Tech Stack

- **Frontend:** React 19, Vite, custom SVG charts (no UI framework)
- **Backend:** FastAPI, PyTorch, scikit-learn, SciPy, WFDB
- **Training:** PyTorch on Google Colab

## Project Structure

```
├── backend/
│   ├── main.py               # FastAPI routes
│   ├── inference.py          # model loading, file parsing, beat segmentation
│   ├── model_definition.py   # ECG_CNN_LSTM_Deep architecture
│   ├── performance.py        # test-set metrics
│   └── requirements.txt
├── frontend/                 # React + Vite web app
├── models/                   # best_model.pth, scaler.pkl
├── examples/                 # beat batches and MIT-BIH records
├── test_samples_csv/         # 100 single-beat test samples
├── results/                  # training figures
├── docs/                     # screenshots
├── ecg(2).ipynb              # training notebook
└── start.bat                 # starts backend + frontend on Windows
```

## Getting Started

**Requirements:** Python 3.10–3.13 and Node.js 18+

```bash
git clone https://github.com/nikhilkumar905/ECG-Arrhythmia-Detection.git
cd ECG-Arrhythmia-Detection

# Backend (http://localhost:8001)
cd backend
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8001

# Frontend (http://localhost:5173), in a second terminal
cd frontend
npm install
npm run dev
```

On Windows, after the first install you can simply run `start.bat`.
Interactive API docs are available at http://localhost:8001/docs.

## Input Formats

| Type | Format |
|---|---|
| Heartbeats | One beat per row, 187 values. An optional 188th column (0–4) is used as the true label. |
| Continuous ECG | One column (or row) of raw samples; set the sampling rate under *Supported formats* (default 360 Hz). |
| WFDB record | Select the `.hea` and `.dat` files together; add `.atr` to compare with the cardiologist annotations. |

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/info` | Model and class information |
| POST | `/api/predict` | Classify an uploaded file (`files`, optional `sampling_rate`) |
| GET | `/api/samples` | List test samples |
| POST | `/api/samples/{name}/predict` | Classify a test sample |
| GET | `/api/records` | List example MIT-BIH records |
| POST | `/api/records/{name}/predict` | Classify an example record |
| GET | `/api/performance` | Test-set metrics |

## Limitations

- The Kaggle split separates heartbeats, not patients, so test scores are higher than for unseen patients.
- The model sees one lead and one beat at a time, without rhythm context.
- Supraventricular and Fusion beats are the hardest classes.

> **Disclaimer:** for research and educational use only. This is not a medical device.

## Acknowledgements

- MIT-BIH Arrhythmia Database: Moody & Mark, *IEEE Eng. Med. Biol.* 20(3), 2001, via
  [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) (Goldberger et al., *Circulation* 101(23), 2000).
- Pre-segmented heartbeat dataset: Kachuee, Fazeli & Sarrafzadeh, *IEEE ICHI* 2018
  ([Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)).

## License

Released under the [MIT License](LICENSE).
