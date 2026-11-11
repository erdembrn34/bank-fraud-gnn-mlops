import os
from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from train_gnn import BankFraudGAT
from drift_monitor import DriftMonitor
from explainability import get_node_feature_importance

app = FastAPI(title="Enterprise Secure GAT Fraud & Risk API")

API_KEY = "banka-gizli-api-anahtari-2026"
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Yetkisiz Erişim: Geçersiz veya Eksik API Key!")

MODEL_PATH = "gnn_fraud_model.pt"
GRAPH_PATH = "banking_graph.pt"

if not os.path.exists(MODEL_PATH) or not os.path.exists(GRAPH_PATH):
    raise FileNotFoundError("Model veya Graph dosyası bulunamadı. Lütfen önce eğitimi çalıştırın.")

model = BankFraudGAT(in_channels=3, hidden_channels=16, out_channels=2)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
model.eval()

data = torch.load(GRAPH_PATH, weights_only=False)

# Drift Monitor Başlatma
drift_monitor = DriftMonitor(baseline_features=data.x)

class FraudCheckRequest(BaseModel):
    node_id: int

def log_request_background(feature_vector: list):
    # Asenkron Arka Plan Görevi: İstekleri drift analizi için loglar
    drift_monitor.log_request(feature_vector)

@app.get("/")
def read_root():
    return {"status": "Active", "system": "Enterprise GAT Fraud Engine"}

@app.post("/predict")
def predict_fraud(
    request: FraudCheckRequest, 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    if request.node_id < 0 or request.node_id >= data.x.shape[0]:
        raise HTTPException(status_code=400, detail="Geçersiz Müşteri / Node ID")

    node_features = data.x[request.node_id].tolist()
    
    # Asenkron olarak drift log kaydı atama
    background_tasks.add_task(log_request_background, node_features)

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        fraud_prob = probs[request.node_id][1].item()
        prediction = torch.argmax(probs[request.node_id]).item()

    # Model Açıklanabilirliği (XAI)
    explanation = get_node_feature_importance(model, data, request.node_id)

    return {
        "node_id": request.node_id,
        "fraud_risk_score": round(fraud_prob, 4),
        "is_suspicious": bool(prediction == 1),
        "feature_importance": explanation
    }

@app.get("/drift-status")
def get_drift_status(api_key: str = Depends(get_api_key)):
    return drift_monitor.check_drift()