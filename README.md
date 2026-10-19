# Bengaluru House Price Prediction — Vercel version

This package adapts the supplied Streamlit app to FastAPI for Vercel and includes a browser interface.

## Features
- Location dropdown populated from every location in `app_options.json` (currently 1,291).
- Area Type dropdown populated from `app_options.json`.
- Prediction uses the supplied trained model pipeline.
- API endpoints: `/api/health`, `/api/options`, `/api/predict`; interactive API documentation at `/docs`.

## Deploy to Vercel
1. Extract the ZIP to a folder on your computer.
2. Create a new GitHub repository and upload all files from the extracted folder, including the model file and `app_options.json`.
3. In Vercel, select **Add New → Project**, import that GitHub repository, and deploy.
4. Keep Root Directory as `./` (repository root). Let Vercel detect the Python/FastAPI app. Do not set a Node build command or output directory.
5. When deployment finishes, test the main website, `/api/health`, and `/docs`.

The original `app.py` was a Streamlit app; this package uses FastAPI because Vercel supports Python/FastAPI functions rather than running a persistent Streamlit server.

## Run locally (Python 3.12 recommended)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```
Open `http://127.0.0.1:8000`.

Predictions are estimates, not guaranteed selling prices.
