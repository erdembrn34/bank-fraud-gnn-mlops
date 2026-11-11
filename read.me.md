# Enterprise Graph Neural Network (GAT) Fraud & Risk Analytics Platform

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-GAT-orange?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Security%20%26%20Async-009688?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)

Bu proje, bankacılık ve finansal ağlar üzerindeki şüpheli para transferlerini (Fraud) ve kara para aklama (AML) şebekelerini **Graph Attention Networks (GAT)** mimarisi kullanarak tespit eden, uçtan uca **MLOps** prensiplerine göre tasarlanmış kurumsal seviyede bir yapay zeka platformudur.

---

## Proje Mimarisi

Sistem, veri üretiminden canlı model izlemeye kadar 6 temel katmandan oluşur:

```text
[ Graph Data ] ──► [ PyTorch Geometric (GAT) ] ──► [ MLflow Experiment Tracking ]
                                                             │
[ Streamlit UI + PyVis ] ◄── [ FastAPI Security & BackgroundTasks ] ◄── [ Drift Monitor (KS-Test) & XAI ]
```

1. **Graph Neural Networks (GAT):** Müşteri transfer ağından `GATConv` katmanları ile düğüm ve ilişki ağırlıklarını (attention weights) öğrenerek fraud tahmini üretir.
2. **Deney Takibi (MLflow):** Eğitim parametrelerini (Learning Rate, Epoch, Loss) ve model ağırlıklarını (`.pt`) versiyonlayarak saklar.
3. **Servis Güvenliği & Asenkron Mimari (FastAPI):** `X-API-Key` yetkilendirmesi ile korunan REST API endpoint'i. `BackgroundTasks` kullanarak canlı istekleri asenkron olarak loglar.
4. **Model Açıklanabilirliği (XAI):** Gradient tabanlı duyarlılık (Feature Importance) analizi ile kararların şeffaf gerekçelerini (Ort. Transfer Tutarı, İşlem Sıklığı, Hesap Yaşı) üretir.
5. **Veri Kayması İzleme (Data Drift):** Canlı API istekleri ile eğitim verisini **Kolmogorov-Smirnov (KS-Testi)** istatistiği ile kıyaslayıp dağılım kaymalarını izler.
6. **İnteraktif Dashboard (Streamlit & PyVis):** Para transfer ağını interaktif HTML haritası üzerinde görselleştirir, canlı risk skorlama ve drift durum takibi sunar.

---

## Teknolojiler ve Kütüphaneler

- **Deep Learning & Graph:** `PyTorch`, `PyTorch Geometric`
- **MLOps & Tracking:** `MLflow`
- **API & Backend:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Data Drift & Analytics:** `Scipy (scipy.stats.ks_2samp)`, `Pandas`, `NumPy`, `Scikit-learn`
- **Visualization & Frontend:** `Streamlit`, `PyVis`, `NetworkX`

---

## Kurulum ve Çalıştırma Adımları

### 1. Ortam Kurulumu

Proje dizinine geçin ve sanal ortamı oluşturup aktifleştirin:

```bash
# Sanal ortam oluşturma
python -m venv venv

# Sanal ortamı aktifleştirme (Windows)
.\venv\Scripts\activate
# Linux / macOS için: source venv/bin/activate

# Bağımlılıkları yükleme
pip install -r requirements.txt
```

### 2. Sentetik Veri Üretimi ve GAT Model Eğitimi

```bash
# 1. Finansal müşteri/transfer ağ verisini üretin
python generate_data.py

# 2. GAT modelini eğitin ve MLflow'a kaydedin
python train_gnn.py
```

### 3. API Servis Katmanını Başlatma (FastAPI)

```bash
uvicorn app:app --reload --port 8000
```

*Servis `http://127.0.0.1:8000` adresinde çalışacaktır.*

### 4. İzleme Panelini Başlatma (Streamlit)

Ayrı bir terminal penceresinde dashboard'u çalıştırın:

```bash
streamlit run dashboard.py
```

*Arayüz `http://localhost:8501` adresinde otomatik olarak açılacaktır.*

*(Opsiyonel)* MLflow deney takibi arayüzünü incelemek için ayrı bir terminalde `mlflow ui` çalıştırıp `http://localhost:5000` adresine gidebilirsiniz.

---

## API Endpoints ve Güvenlik Yapısı

API üzerindeki tüm tahmin ve izleme isteklerinde `X-API-Key` başlığı zorunludur:
`X-API-Key: banka-gizli-api-anahtari-2026`

| Endpoint | Method | Açıklama |
| :--- | :--- | :--- |
| `/` | `GET` | Health Check (Servis durum kontrolü) |
| `/predict` | `POST` | Belirtilen `node_id` için fraud risk skoru, sınıflandırma ve XAI gerekçesini döner |
| `/drift-status` | `GET` | Canlı veriler ile baseline verisi arasındaki Kolmogorov-Smirnov test sonuçlarını döner |

---

## Örnek Çıktılar

### 1. Model Tahmini ve XAI (Karar Gerekçesi) API Yanıtı
```json
{
  "node_id": 12,
  "fraud_risk_score": 0.8421,
  "is_suspicious": true,
  "feature_importance": {
    "Ort. Transfer Tutarı": 0.4512,
    "İşlem Sıklığı": 0.3891,
    "Hesap Yaşı": 0.1597
  }
}
```

### 2. Data Drift Test Çıktısı (Kolmogorov-Smirnov Testi)
```json
{
  "status": "STABLE",
  "feature_details": {
    "feature_0": { "ks_stat": 0.2583, "p_value": 0.3097, "drift_detected": false },
    "feature_1": { "ks_stat": 0.2102, "p_value": 0.5595, "drift_detected": false },
    "feature_2": { "ks_stat": 0.1554, "p_value": 0.8732, "drift_detected": false }
  }
}
```