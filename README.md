# Deep Learning-Based Predictive Maintenance

An end-to-end industrial predictive maintenance web application that estimates Remaining Useful Life (RUL) of turbofan jet engines from NASA C-MAPSS simulation data using a PyTorch LSTM model, a high-performance FastAPI backend, and an interactive React dashboard.

> **Safety & Research Disclaimer:**  
> The NASA C-MAPSS dataset is simulated run-to-failure benchmark data. This application is an engineering demonstration prototype for predictive modeling research and education. It is **not** certified aviation maintenance software and must **never** be used for operational flight decisions or maintenance dispatching.

---

## 1. System Architecture

The project is structured into three integrated tiers:

```
project-b-predictive-maintenance/
├── data/                  # NASA C-MAPSS whitespace-delimited raw trajectory files
├── artifacts/             # Trained PyTorch weights (.pt), scalers (.joblib), configs & metrics
├── reports/               # Training logs, test prediction CSVs, and baseline comparison tables
├── src/                   # Core Machine Learning & Data Pipeline
│   ├── data.py            # Whitespace parsing, RUL labeling, sliding window generators
│   ├── model.py           # PyTorch LSTM regressor architecture (RULLSTM)
│   ├── train.py           # Trajectory-grouped training, early stopping, NASA scoring
│   ├── evaluate.py        # Benchmark evaluation & naive/regression baseline comparison
│   └── inference.py       # Artifact loading, preprocessing, inference & health classification
├── backend/               # FastAPI Inference & Telemetry Service
│   ├── main.py            # REST endpoints, CORS middleware, error handling (400, 404, 422, 503)
│   └── schemas.py         # Pydantic v2 validation models
├── frontend/              # Modern React + Vite Industrial Dashboard
│   ├── src/App.jsx        # Complete dashboard: metrics, gauges, dual charts, fleet table
│   ├── src/style.css      # Aerospace dark theme, responsive grids, SVG animations
│   └── vite.config.js     # React build config with standard port bindings
└── tests/                 # Comprehensive unittest suite (29 tests across all tiers)
```

---

## 2. NASA C-MAPSS Dataset Setup

The NASA Prognostics Center of Excellence (PCoE) Turbofan Engine Degradation Simulation Dataset consists of 4 operating subsets (`FD001`, `FD002`, `FD003`, `FD004`). Each subset contains:
- `train_FD00x.txt`: Multi-engine run-to-failure trajectories.
- `test_FD00x.txt`: Multi-engine operational histories cut off prior to failure.
- `RUL_FD00x.txt`: Ground truth remaining useful life at the final cycle of each test engine.

### Data Format
Files are whitespace-separated with 26 columns per observation:
- `engine_id`: Unit identifier (integer).
- `cycle`: Operating time in cycles (integer).
- `setting_1`, `setting_2`, `setting_3`: Flight condition operating settings.
- `s1` – `s21`: 21 turbofan sensor channels (temperatures, pressures, fan speeds, bleed rates).

### Automatic Validation & Preprocessing
- **RUL Labeling**: Piecewise constant linear degradation function with configurable clipping cap (default: $RUL \le 125$ cycles).
- **Leakage Prevention**: Standard scalers and constant feature detectors are fit **only** on training engine trajectories.
- **Sliding Windows**: Fixed-length temporal sequences (default: 30 cycles). Short trajectories are left-padded by repeating the earliest observed telemetry state.

---

## 3. Environment & Installation

### Prerequisites
- Python 3.10 – 3.14
- Node.js 18+ and npm
- Windows PowerShell (or macOS / Linux bash)

### Windows PowerShell Setup

Run commands from the project root:

```powershell
# 1. Create and activate Python virtual environment
py -m venv .venv

# If script execution is restricted on Windows, allow process-level bypass:
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install Python dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend
npm ci
cd ..
```

---

## 4. Model Training & Evaluation

### Train the LSTM Regressor
To train the default `FD003` model from scratch:

```powershell
python -m src.train --subset FD003
```

#### Configurable Training Hyperparameters:
```powershell
python -m src.train `
  --subset FD003 `
  --sequence-length 30 `
  --hidden-size 64 `
  --num-layers 2 `
  --dropout 0.15 `
  --batch-size 64 `
  --learning-rate 0.001 `
  --epochs 25 `
  --patience 5 `
  --seed 42 `
  --rul-cap 125
```

### Generated Artifacts
Training automatically saves:
- `artifacts/model_fd003.pt`: Best checkpoint weights and model state dict.
- `artifacts/scaler_fd003.joblib`: Fitted `StandardScaler`.
- `artifacts/config_fd003.json`: Feature column order, sequence window length, and status thresholds.
- `artifacts/metrics_fd003.json`: Official test MAE, RMSE, and asymmetric NASA score.
- `reports/training_history_fd003.csv`: Epoch-by-epoch loss tracking.
- `reports/test_predictions_fd003.csv`: Predictions vs. ground truth for all 100 test engines.

### Official Test Set Benchmark Comparison

Run the independent evaluation and baseline benchmarking module:

```powershell
python -m src.evaluate --subset FD003
```

#### Official Evaluation Results (FD003 Test Split):
| Model Architecture | Test MAE (cycles) | Test RMSE (cycles) | NASA PHM Score | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch LSTM Regressor (Ours)** | **11.32** | **15.11** | **388.0** | **0.8668** |
| Naive Mean Baseline | 38.00 | 45.07 | 57,949.6 | -0.1854 |
| Naive Median Baseline | 51.42 | 63.14 | 1,138,771.6 | -1.3267 |
| Ridge Regressor Baseline | 75.32 | 85.95 | 709,176.7 | -3.3106 |

*The NASA PHM asymmetric score heavily penalizes late (optimistic) failure predictions compared to early (conservative) predictions.*

---

## 5. Running the Application

For live execution, open **two separate PowerShell terminals**:

### Terminal 1: FastAPI Backend
```powershell
cd project-b-predictive-maintenance
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### Terminal 2: React Dashboard (Frontend)
```powershell
cd project-b-predictive-maintenance\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
- Dashboard URL: `http://127.0.0.1:5173`

### Production Frontend Build
To verify and compile the optimized production bundle:
```powershell
cd project-b-predictive-maintenance\frontend
npm run build
```
*(Builds into `frontend/dist/` in under 3 seconds with zero errors).*

---

## 6. REST API Reference

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Check API, dataset, and model readiness | `200` |
| `GET` | `/api/subsets` | List supported C-MAPSS subsets & training state | `200` |
| `GET` | `/api/metrics?subset=FD003` | Retrieve official benchmark metrics & baselines | `200`, `400`, `404` |
| `GET` | `/api/engines?subset=FD003` | List test engines with degradation status summary | `200`, `400` |
| `GET` | `/api/engine/{id}?subset=FD003` | Full cycle-by-cycle telemetry history & ground truth | `200`, `400`, `404`, `422` |
| `GET` | `/api/engine/{id}/predict` | Predict RUL & status for a specific test engine | `200`, `400`, `404`, `503` |
| `POST` | `/api/predict` | Predict RUL for an engine ID or raw sensor sequence | `200`, `400`, `404`, `503` |

### Health Status Thresholds
Predictions are categorized using industrial maintenance thresholds:
- **Normal**: $\text{RUL} > 50$ cycles.
- **Warning**: $20 < \text{RUL} \le 50$ cycles (inspection recommended).
- **Critical**: $\text{RUL} \le 20$ cycles (immediate service required).

---

## 7. Automated Test Suite

The project includes 29 unit and integration tests covering the data pipeline, model layers, evaluation math, inference reloading, and all FastAPI endpoints:

```powershell
python -m unittest discover -s tests -v
```

### Test Coverage Breakdown:
- `tests/test_data.py`: Whitespace parsing, column alignment, RUL label clipping, left-padding window generation, scaler isolation.
- `tests/test_model.py`: Layer initialization, invalid parameter rejection, forward batch shapes, gradient backpropagation.
- `tests/test_evaluate.py`: Asymmetric NASA score calculation, zero-error edge cases, optimistic vs pessimistic penalty checks.
- `tests/test_inference.py`: Threshold status assignment, output dimensions, non-negative clipping, and saved artifact bundle reloading.
- `tests/test_backend.py`: Health endpoints, subsets list, metrics schema, engine history, single-engine prediction, raw sequence prediction, input validation, untrained subset handling (503), missing engine handling (404), and CORS headers.

---

## 8. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: train_FD003.txt` | Dataset files missing from `data/` folder. | Ensure `train_FD003.txt`, `test_FD003.txt`, and `RUL_FD003.txt` are in `data/`. |
| `503: No trained model bundle available` | Model has not been trained for the requested subset. | Run `python -m src.train --subset <SUBSET>` to produce artifacts. |
| `Failed to fetch` on Frontend | Backend is not running or port 8000 is blocked. | Verify `python -m uvicorn backend.main:app` is running and check `frontend/.env`. |
| PowerShell script execution error | Execution policy is restricted on Windows. | Run `Set-ExecutionPolicy -Scope Process Bypass` in the current terminal. |
| Port 8000 or 5173 already in use | Another service is using the port. | Terminate the occupying process or pass `--port <new_port>` to uvicorn/vite. |

---

## 9. Known Limitations

1. **Simulated Physics**: C-MAPSS data is derived from commercial modular aero-propulsion simulation models and does not represent all real-world environmental contamination or variable degradation dynamics.
2. **Deterministic Point Estimates**: Current predictions are point estimates; future work includes implementing Monte Carlo Dropout or Bayesian LSTMs for prediction uncertainty intervals.
3. **Padded Sequences**: Engines with fewer observed cycles than the window length (30) use left-repetition padding.
