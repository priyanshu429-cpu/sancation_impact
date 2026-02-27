import requests
import torch
from torch_geometric.data import Data

# -----------------------------------
# Countries in graph
# -----------------------------------
COUNTRIES = ["India", "USA", "China", "Russia", "Germany"]
country_to_idx = {c: i for i, c in enumerate(COUNTRIES)}

COUNTRY_CODES = {
    "India": "IND",
    "USA": "USA",
    "China": "CHN",
    "Russia": "RUS",
    "Germany": "DEU"
}

# -----------------------------------
# 8 Economic Indicators
# -----------------------------------
INDICATORS = [
    "NY.GDP.MKTP.CD",       # GDP
    "NE.EXP.GNFS.ZS",       # exports
    "NE.IMP.GNFS.ZS",       # imports
    "FP.CPI.TOTL.ZG",       # inflation
    "FI.RES.TOTL.CD",       # reserves
    "BX.KLT.DINV.CD.WD",    # FDI
    "EG.USE.PCAP.KG.OE",    # energy use
    "SP.POP.TOTL"           # population
]


# -----------------------------------
# Fetch and Normalize Node Features
# -----------------------------------
def fetch_country_features():

    print("Downloading economic indicators...")

    features = torch.zeros(len(COUNTRIES), len(INDICATORS))

    for country, code in COUNTRY_CODES.items():
        idx = country_to_idx[country]
        values = []

        for ind in INDICATORS:
            try:
                url = f"https://api.worldbank.org/v2/country/{code}/indicator/{ind}?format=json&per_page=1"
                r = requests.get(url, timeout=15)
                data = r.json()

                value = data[1][0]["value"]
                if value is None:
                    value = 0
            except:
                value = 0

            values.append(float(value))

        features[idx] = torch.tensor(values)

    # -------------------------------
    # FIX: Normalize economic features
    # -------------------------------

    # Log scale (handles trillions safely)
    features = torch.log1p(torch.abs(features))

    # Standardization (mean 0, std 1)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-6
    features = (features - mean) / std

    print("Feature normalization complete.")

    return features


# -----------------------------------
# Static Trade Network (No IMF API)
# -----------------------------------
def fetch_trade_edges():

    print("Using predefined global trade network...")

    # Order: India, USA, China, Russia, Germany
    trade_matrix = torch.tensor([
        [0,   80, 120,  30,  60],   # India
        [70,   0, 200,  40, 150],   # USA
        [110,180,   0,  90, 160],   # China
        [35,  45, 100,   0,  80],   # Russia
        [65, 140,170,  85,   0]     # Germany
    ], dtype=torch.float32)

    edge_list = []
    weights = []

    n = trade_matrix.size(0)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            weight = trade_matrix[i][j]

            if weight > 0:
                edge_list.append([i, j])
                weights.append(weight / 200.0)  # normalize to ~0–1

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    print("Total edges created:", len(edge_list))

    return edge_index, edge_weight


# -----------------------------------
# Load STATIC Graph (Used in Training & Inference)
# -----------------------------------
def load_trade_graph(policy_vector):

    data = torch.load("trade_graph.pt")

    x = data["x"].clone()
    edge_index = data["edge_index"]
    edge_weight = data["edge_weight"]

    # Append sanction policy to each country
    policy_expand = policy_vector.repeat(x.size(0), 1)
    x = torch.cat([x, policy_expand], dim=1)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)