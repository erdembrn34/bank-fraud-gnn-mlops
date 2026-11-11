import streamlit as st
import torch
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
import pandas as pd

st.set_page_config(page_title="Enterprise Fraud GAT Monitor", layout="wide")
st.title(" Finansal Ağ Anomali & Risk Analiz Platformu (GAT + XAI + MLOps)")

data = torch.load("banking_graph.pt", weights_only=False)

st.sidebar.header(" Müşteri / Node Sorgulama")
node_id = st.sidebar.number_input("Node ID Seçin", min_value=0, max_value=data.x.shape[0]-1, value=12)

headers = {"X-API-Key": "banka-gizli-api-anahtari-2026"}

if st.sidebar.button("Risk Skorunu Sorgula"):
    try:
        response = requests.post("http://localhost:8000/predict", json={"node_id": node_id}, headers=headers)
        
        if response.status_code == 200:
            res_data = response.json()
            st.sidebar.subheader("API Tahmin Sonucu:")
            st.sidebar.metric("Fraud Risk Skoru", f"%{res_data['fraud_risk_score']*100:.2f}")
            
            if res_data['is_suspicious']:
                st.sidebar.error(" YÜKSEK RİSK: Şüpheli İşlem Ağında!")
            else:
                st.sidebar.success(" NORMAL: Düşük Risk Seviyesi")
                
            st.sidebar.subheader(" Model Karar Gerekçesi (XAI):")
            importance_df = pd.DataFrame(list(res_data['feature_importance'].items()), columns=['Özellik', 'Önem Oranı'])
            st.sidebar.bar_chart(importance_df.set_index('Özellik'))
        else:
            st.sidebar.error(f"Erişim Engellendi (HTTP {response.status_code})")
    except Exception:
        st.sidebar.warning("FastAPI servisine ulaşılamadı.")

# Data Drift Kontrol Butonu
if st.sidebar.button("Canlı Veri Kayması (Drift) Kontrolü"):
    try:
        drift_res = requests.get("http://localhost:8000/drift-status", headers=headers).json()
        st.sidebar.subheader("Data Drift Durumu:")
        st.sidebar.json(drift_res)
    except Exception:
        st.sidebar.warning("Drift servisine ulaşılamadı.")

st.subheader(" Müşteri Para Transfer Ağ Haritası (PyVis)")

net = Network(height="500px", width="100%")
net.bgcolor = "#222222"
net.font_color = "white"

for i in range(100):
    is_target = (i == node_id)
    color = "gold" if is_target else ("red" if data.y[i].item() == 1 else "#00ff1e")
    size = 25 if is_target else 10
    net.add_node(i, label=f"Müşteri {i}", color=color, size=size)

edge_src = data.edge_index[0].numpy()
edge_dst = data.edge_index[1].numpy()

for src, dst in zip(edge_src, edge_dst):
    if src < 100 and dst < 100:
        net.add_edge(int(src), int(dst))

net.save_graph("network.html")

with open("network.html", "r", encoding="utf-8") as f:
    source_code = f.read()

components.html(source_code, height=520)