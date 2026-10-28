# Real Estate Price Predictor

A Streamlit application for estimating property prices and exploring real-estate
market information for Chhatrapati Sambhajinagar.

## Requirements

- Python 3.10 or newer
- The dataset at `data/real_data1.csv`

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_app.py
```

Open the local URL printed by the launcher. On a fresh database, create an
account on the Register tab; the application no longer seeds a default
account. Existing local accounts are preserved, and legacy password hashes
are upgraded when their owners log in.

The model is trained from `data/real_data1.csv` when it is missing and saved to
`models/real_estate_model.pkl`. User accounts and property history are stored
in `data/real_estate_app.db`. These generated files are intentionally excluded
from Git.

## Docker

```powershell
docker build -t real-estate-price-predictor .
docker run --rm -p 8000:8000 real-estate-price-predictor
```

Then open `http://localhost:8000`.

## Project layout

- `src/app.py` — Streamlit application
- `src/train_model.py` — model training pipeline
- `src/generate_sample_data.py` — generates the supplementary sample CSV
- `data/real_data1.csv` — training and application dataset
- `infra/` — Azure Container Apps deployment template

## Notes before deployment

The SQLite database is local to the container unless persistent storage is
configured. The included Azure template is a starting point; set a real
container image in `infra/main.parameters.json` before deploying.
