from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from macro_risk_model import MacroRiskModel
from model import SanctionImpactGNN
from utils import load_trade_graph
from fastapi import Query
# ==============================
# Initialize API
# ==============================

app = FastAPI(title="Sanction Impact Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
risk_model = MacroRiskModel()
risk_model.load_state_dict(torch.load("macro_risk_model.pt", map_location="cpu"))
risk_model.eval()
# ==============================
# Load Models
# ==============================

print("Loading GNN...")
gnn_model = SanctionImpactGNN(in_dim=15)
gnn_model.load_state_dict(torch.load("saved_model.pt", map_location="cpu"))
gnn_model.eval()
print("GNN ready.")

print("Loading NLP model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
nlp_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
print("NLP ready.")

# ==============================
# Severity Model
# ==============================

def compute_severity(financial, trade, technology, energy, issuer_strength, binding):
    severity = (
        0.25 * financial +
        0.20 * trade +
        0.15 * technology +
        0.20 * energy +
        0.10 * issuer_strength +
        0.10 * binding
    )
    return min(1.0, severity)

# ==============================
# Schemas
# ==============================

class PolicyInput(BaseModel):
    financial: int
    trade: int
    technology: int
    energy: int
    issuer_strength: float
    binding: int

class ExplanationInput(BaseModel):
    metric: str
    value: float
    context: dict

# ==============================
# Prediction Endpoint
# ==============================

@app.post("/predict")
def predict(policy: PolicyInput):
    try:
        severity = compute_severity(
            policy.financial,
            policy.trade,
            policy.technology,
            policy.energy,
            policy.issuer_strength,
            policy.binding
        )

        policy_vector = torch.tensor([
            severity,
            policy.financial,
            policy.trade,
            policy.technology,
            policy.energy,
            policy.issuer_strength,
            policy.binding
        ], dtype=torch.float32)

        graphs = [load_trade_graph(policy_vector) for _ in range(5)]

        with torch.no_grad():
            preds = gnn_model(graphs)

        return {
            "severity": severity,
            "gdp": float(preds["gdp"].item()),
            "trade": float(preds["trade"].item()),
            "fdi": float(preds["fdi"].item())
        }

    except Exception as e:
        return {"error": str(e)}

# ==============================
# NLP Explanation
# ==============================

@app.post("/explain")
def explain_metric(data: ExplanationInput):
    try:
        prompt = f"""
You are an economic analyst with light office-style humor.

Country: {data.context.get('country', 'the selected country')}

Explain why {data.metric.upper()} changed for {data.context.get('country', 'this country')}.

Observed value: {data.value:.2f}

Important:
- Focus ONLY on the specified country.
- Do not default to the United States unless the country is USA.
- Keep explanation grounded in macroeconomic logic.

Provide a concise explanation (4–6 sentences).
"""

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = nlp_model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            early_stopping=True,
        )

        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"explanation": explanation}

    except Exception as e:
        return {"error": str(e)}

# ==============================
# Model-driven Risk Endpoint
# ==============================

@app.get("/risk")
def get_risk():

    countries = [
        "Russia","China","USA","Iran","North Korea","EU","UAE","Saudi Arabia",
        "Japan","South Korea","Turkey","Israel","Pakistan","UK","Singapore",
        "Australia","Germany","France","Brazil","South Africa"
    ]

    results = []

    for country in countries:

        # Neutral policy scenario
        severity = compute_severity(1,1,0,1,0.7,1)

        policy_vector = torch.tensor([
            severity, 1, 1, 0, 1, 0.7, 1
        ], dtype=torch.float32)

        graphs = [load_trade_graph(policy_vector) for _ in range(5)]

        with torch.no_grad():
            preds = gnn_model(graphs)

        gdp = float(preds["gdp"].item())
        trade = float(preds["trade"].item())
        fdi = float(preds["fdi"].item())

        shock = abs(gdp) + abs(trade) + abs(fdi)

        if shock > 8:
            level = "critical"
        elif shock > 5:
            level = "high"
        elif shock > 2:
            level = "medium"
        else:
            level = "low"

        compliance = int((1 - min(shock / 10, 1)) * 100)

        results.append({
            "country": country,
            "riskLevel": level,
            "sanctionCount": int(shock * 10),
            "complianceScore": compliance,
            "impactOnIndia": f"Model-derived shock score {shock:.2f}"
        })

    return results

# ==============================
# macro risk model
# ==============================

from pydantic import BaseModel
import requests
import torch

class CountryInput(BaseModel):
    country_code: str

def get_latest_indicator(country_code, indicator_code):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&per_page=500"
    r = requests.get(url)
    data = r.json()

    if not isinstance(data, list) or len(data) < 2:
        return 0

    for entry in data[1]:
        if entry.get("value") is not None:
            return entry["value"]

    return 0


@app.post("/macro-risk")
def macro_risk(data: CountryInput):

    gdp = get_latest_indicator(data.country_code, "NY.GDP.PCAP.CD") or 0
    trade = get_latest_indicator(data.country_code, "NE.TRD.GNFS.ZS") or 0
    fdi = get_latest_indicator(data.country_code, "BX.KLT.DINV.WD.GD.ZS") or 0
    energy = get_latest_indicator(data.country_code, "EG.IMP.CONS.ZS") or 0

    # simple scaling (same logic as training — improve later)
    gdp = min(gdp / 100000, 1)
    trade = min(trade / 100, 1)
    fdi = min(abs(fdi) / 20, 1)
    energy = min(energy / 100, 1)

    features = torch.tensor([[gdp, trade, fdi, energy]], dtype=torch.float32)

    with torch.no_grad():
        risk_score = float(risk_model(features).item())

    return {
        "country": data.country_code,
        "risk_score": risk_score
    }

# ==============================
# Time-series endpoints
# ==============================
@app.get("/macro-timeseries")
def macro_timeseries(country_code: str = Query(...)):

    def get_series(indicator):
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=100"
        r = requests.get(url)
        data = r.json()

        if not isinstance(data, list) or len(data) < 2:
            return []

        series = []
        for entry in data[1]:
            if entry.get("value") is not None:
                series.append({
                    "year": entry["date"],
                    "value": entry["value"]
                })

        return series[:10]  # last 10 years

    return {
        "gdp": get_series("NY.GDP.MKTP.KD.ZG"),       # GDP growth
        "trade": get_series("NE.TRD.GNFS.ZS"),        # Trade % GDP
        "fdi": get_series("BX.KLT.DINV.WD.GD.ZS"),    # FDI % GDP
    }