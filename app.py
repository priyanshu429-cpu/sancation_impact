from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import torch
import os
import requests

from macro_risk_model import MacroRiskModel
from model import SanctionImpactGNN
from utils import load_trade_graph

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

# ==============================
# SAFE MODEL LOADING
# ==============================

risk_model = None
gnn_model = None


def load_risk_model():
    global risk_model
    if risk_model is None:
        print("Loading Macro Risk Model...")
        if not os.path.exists("macro_risk_model.pt"):
            print("WARNING: macro_risk_model.pt not found")
            return None

        risk_model = MacroRiskModel()
        risk_model.load_state_dict(torch.load("macro_risk_model.pt", map_location="cpu"))
        risk_model.eval()
    return risk_model


def load_gnn_model():
    global gnn_model
    if gnn_model is None:
        print("Loading GNN model...")
        if not os.path.exists("saved_model.pt"):
            print("WARNING: saved_model.pt not found")
            return None

        gnn_model = SanctionImpactGNN(in_dim=15)
        gnn_model.load_state_dict(torch.load("saved_model.pt", map_location="cpu"))
        gnn_model.eval()
    return gnn_model


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


class CountryInput(BaseModel):
    country_code: str


# ==============================
# Prediction Endpoint
# ==============================

@app.post("/predict")
def predict(policy: PolicyInput):

    gnn = load_gnn_model()
    if gnn is None:
        return {"error": "GNN model weights not found on server"}

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
        preds = gnn(graphs)

    return {
        "severity": severity,
        "gdp": float(preds["gdp"].item()),
        "trade": float(preds["trade"].item()),
        "fdi": float(preds["fdi"].item())
    }


# ==============================
# NLP Explanation (FIXED LIGHTWEIGHT)
# ==============================

@app.post("/explain")
async def explain_metric(data: dict = Body(...)):
    try:
        metric = data.get("metric")
        value = data.get("value")
        context = data.get("context", {})
        country = context.get("country", "this country")

        if metric == "gdp":
            if value < 0:
                explanation = (
                    f"{country}'s GDP contraction indicates economic slowdown. "
                    "Sanctions likely reduced trade flows and foreign investments."
                )
            elif value < 3:
                explanation = (
                    f"{country}'s GDP growth is moderate. Sanctions may be limiting expansion."
                )
            else:
                explanation = (
                    f"{country} shows strong GDP growth despite sanctions, indicating resilience."
                )

        elif metric == "trade":
            explanation = (
                f"Trade openness at {value}% reflects exposure to global markets. "
                "Sanctions can disrupt imports and exports."
            )

        elif metric == "fdi":
            explanation = (
                f"FDI trend of {value} billion USD reflects investor confidence. "
                "Sanctions often reduce foreign investment."
            )

        else:
            explanation = f"{metric} value is {value}, reflecting economic conditions."

        return {"explanation": explanation}

    except Exception as e:
        print("Explain error:", e)
        return {"explanation": "Error generating explanation"}


# ==============================
# Macro Risk Endpoint
# ==============================

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

    model = load_risk_model()
    if model is None:
        return {"error": "macro risk model weights missing"}

    gdp = get_latest_indicator(data.country_code, "NY.GDP.PCAP.CD") or 0
    trade = get_latest_indicator(data.country_code, "NE.TRD.GNFS.ZS") or 0
    fdi = get_latest_indicator(data.country_code, "BX.KLT.DINV.WD.GD.ZS") or 0
    energy = get_latest_indicator(data.country_code, "EG.IMP.CONS.ZS") or 0

    gdp = min(gdp / 100000, 1)
    trade = min(trade / 100, 1)
    fdi = min(abs(fdi) / 20, 1)
    energy = min(energy / 100, 1)

    features = torch.tensor([[gdp, trade, fdi, energy]], dtype=torch.float32)

    with torch.no_grad():
        risk_score = float(model(features).item())

    return {
        "country": data.country_code,
        "risk_score": risk_score
    }


# ==============================
# Macro Timeseries
# ==============================

@app.get("/macro-timeseries")
def macro_timeseries(country_code: str):
    return [
        {"year": 2018, "gdp": 2.5, "trade": 1.2, "fdi": 0.8},
        {"year": 2019, "gdp": 3.0, "trade": 1.5, "fdi": 1.0},
        {"year": 2020, "gdp": -1.2, "trade": -0.5, "fdi": 0.3},
        {"year": 2021, "gdp": 6.5, "trade": 2.0, "fdi": 1.5},
        {"year": 2022, "gdp": 5.8, "trade": 1.8, "fdi": 1.2},
        {"year": 2023, "gdp": 6.2, "trade": 2.1, "fdi": 1.4},
        {"year": 2024, "gdp": 6.7, "trade": 2.3, "fdi": 1.6},
        {"year": 2025, "gdp": 7.0, "trade": 2.5, "fdi": 1.8},
    ]

# ==============================
# Serve Frontend
# ==============================

@app.get("/")
def serve_frontend():
    return FileResponse("dist/index.html")


app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")