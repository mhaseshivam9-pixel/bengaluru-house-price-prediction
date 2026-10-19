from pathlib import Path
import json
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
with (APP_DIR / "app_options.json").open("r", encoding="utf-8") as f:
    OPTIONS = json.load(f)
LOCATIONS = OPTIONS.get("locations", [])
AREA_TYPES = OPTIONS.get("area_types", [])
if not LOCATIONS or not AREA_TYPES:
    raise RuntimeError("app_options.json must contain locations and area_types.")

model = joblib.load(APP_DIR / "best_house_price_model.joblib")
metadata = joblib.load(APP_DIR / "model_metadata.joblib")
INPUT_FEATURES = metadata["input_features"]

app = FastAPI(title="Bengaluru House Price Prediction API", version="1.0.0")


class HouseDetails(BaseModel):
    location: str
    total_sqft: float = Field(gt=0, le=100000)
    bhk: int = Field(gt=0, le=20)
    bath: int = Field(gt=0, le=20)
    balcony: int = Field(ge=0, le=10)
    area_type: str


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True, "locations_count": len(LOCATIONS)}


@app.get("/api/options")
def get_options():
    return {"locations": LOCATIONS, "area_types": AREA_TYPES}


@app.post("/api/predict")
def predict_house(details: HouseDetails):
    if details.location not in LOCATIONS:
        raise HTTPException(status_code=422, detail="Please choose a location from the dataset dropdown.")
    if details.area_type not in AREA_TYPES:
        raise HTTPException(status_code=422, detail="Please choose a valid area type.")
    house = pd.DataFrame([{
        "location": details.location,
        "total_sqft": details.total_sqft,
        "bhk": details.bhk,
        "bath": details.bath,
        "balcony": details.balcony,
        "area_type": details.area_type,
    }], columns=INPUT_FEATURES)
    try:
        prediction = float(model.predict(house)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Prediction failed. Please check the property details.") from exc
    if not math.isfinite(prediction) or prediction < 0:
        raise HTTPException(status_code=400, detail="The model returned an invalid price estimate.")
    return {"predicted_price_lakhs": round(prediction, 2), "currency": "INR", "unit": "lakhs"}


HTML_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bengaluru House Price Predictor</title><style>
*{box-sizing:border-box}body{margin:0;padding:28px 14px;background:#f3f6fc;color:#172033;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:820px;margin:auto}.hero{background:#17345f;color:white;padding:25px;border-radius:18px;margin-bottom:18px}
h1{margin:0 0 8px;font-size:clamp(1.5rem,4vw,2rem)}.hero p{margin:0;line-height:1.5;color:#e0eaff}
.card{background:white;border:1px solid #e1e7f0;border-radius:16px;padding:24px;box-shadow:0 8px 24px #1720330b}h2{margin-top:0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.field{display:flex;flex-direction:column;gap:7px}.full{grid-column:1/-1}
label{font-weight:650;font-size:.92rem}input,select{width:100%;min-height:46px;padding:11px 12px;border:1px solid #c8d2e2;border-radius:9px;background:white;font:inherit;color:inherit}
input:focus,select:focus{outline:3px solid #bfdbfe;border-color:#2563eb}.hint{font-size:.8rem;color:#64748b}
button{width:100%;margin-top:20px;padding:14px;border:0;border-radius:10px;background:#2563eb;color:white;font:inherit;font-weight:700;cursor:pointer}button:disabled{opacity:.6}
#error{display:none;margin-top:14px;padding:12px;border-radius:9px;background:#fef2f2;color:#991b1b}
#result{display:none;margin-top:18px;padding:18px;border-radius:12px;background:#eff6ff;border:1px solid #bfdbfe}.prices{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.price{background:white;border-radius:10px;padding:14px}.price small{display:block;color:#64748b;margin-bottom:6px}.price strong{color:#1d4ed8;font-size:1.25rem}
footer{text-align:center;color:#64748b;font-size:.82rem;margin:18px 0}@media(max-width:580px){.grid,.prices{grid-template-columns:1fr}.card{padding:18px}}
</style></head><body><main>
<section class="hero"><h1>🏠 Bengaluru House Price Predictor</h1><p>Choose a location from the dataset and enter property details to estimate its price.</p></section>
<section class="card"><h2>Property Details</h2><form id="form"><div class="grid">
<div class="field full"><label for="location">Location</label><select id="location" required><option value="">Loading locations…</option></select><div class="hint" id="count">Locations are loaded from the dataset options.</div></div>
<div class="field"><label for="sqft">Total Area (square feet)</label><input id="sqft" type="number" min="100" max="100000" step="any" value="1200" required></div>
<div class="field"><label for="bhk">BHK</label><input id="bhk" type="number" min="1" max="20" step="1" value="2" required></div>
<div class="field"><label for="bath">Number of Bathrooms</label><input id="bath" type="number" min="1" max="20" step="1" value="2" required></div>
<div class="field"><label for="balcony">Number of Balconies</label><input id="balcony" type="number" min="0" max="10" step="1" value="1" required></div>
<div class="field full"><label for="area">Area Type</label><select id="area" required><option value="">Loading area types…</option></select></div>
</div><button id="submit" type="submit">Predict House Price</button></form>
<div id="error" role="alert"></div><div id="result" aria-live="polite"><h3>Estimated House Price</h3><div class="prices">
<div class="price"><small>Estimated Price (Lakhs)</small><strong id="lakhs">—</strong></div><div class="price"><small>Estimated Price (Rupees)</small><strong id="rupees">—</strong></div></div>
<p class="hint">This is a machine-learning estimate, not a guaranteed selling price or professional valuation.</p></div></section>
<footer>Bengaluru House Price Prediction · Academic machine-learning project</footer></main>
<script>
const loc=document.getElementById('location'), area=document.getElementById('area'), err=document.getElementById('error'), result=document.getElementById('result'), btn=document.getElementById('submit');
const rupee=new Intl.NumberFormat('en-IN',{maximumFractionDigits:0}), lakhsFmt=new Intl.NumberFormat('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});
async function loadOptions(){try{const r=await fetch('/api/options');if(!r.ok)throw Error('Could not load dataset dropdown options.');const d=await r.json();loc.replaceChildren(new Option('Select a location',''));d.locations.forEach(x=>loc.add(new Option(x,x)));area.replaceChildren(new Option('Select area type',''));d.area_types.forEach(x=>area.add(new Option(x,x)));document.getElementById('count').textContent=d.locations.length.toLocaleString('en-IN')+' dataset locations available. Focus the dropdown and type to jump to a location.';}catch(e){err.textContent=e.message;err.style.display='block';}}
document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();err.style.display='none';result.style.display='none';btn.disabled=true;btn.textContent='Predicting…';const payload={location:loc.value,total_sqft:Number(document.getElementById('sqft').value),bhk:Number(document.getElementById('bhk').value),bath:Number(document.getElementById('bath').value),balcony:Number(document.getElementById('balcony').value),area_type:area.value};try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.detail||'Prediction failed. Check the inputs.');document.getElementById('lakhs').textContent='₹'+lakhsFmt.format(d.predicted_price_lakhs)+' lakh';document.getElementById('rupees').textContent='₹'+rupee.format(d.predicted_price_lakhs*100000);result.style.display='block';}catch(e){err.textContent=e.message;err.style.display='block';}finally{btn.disabled=false;btn.textContent='Predict House Price';}});
loadOptions();
</script></body></html>"""
