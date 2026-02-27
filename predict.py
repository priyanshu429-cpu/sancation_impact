import torch
from model import SanctionImpactGNN

# load model
model = SanctionImpactGNN(in_dim=10)  # adjust feature size
model.load_state_dict(torch.load("saved_model.pt"))
model.eval()

def run_prediction(data_list):
    with torch.no_grad():
        outputs = model(data_list)
    return outputs
