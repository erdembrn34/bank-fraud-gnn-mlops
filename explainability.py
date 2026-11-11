import torch

def get_node_feature_importance(model, data, node_id):
    """
    Modelin ilgili düğüm için karar verirken kullandığı feature duyarlılıklarını (Gradients) hesaplar.
    """
    model.eval()
    x = data.x.clone().detach().requires_grad_(True)
    
    out = model(x, data.edge_index)
    target_score = out[node_id, 1]  # Fraud sınıfı skoru
    target_score.backward()
    
    # Gradient büyüklüklerine göre önem derecelendirmesi
    gradients = x.grad[node_id].abs()
    importance_scores = (gradients / gradients.sum()).tolist()
    
    feature_names = ["Ort. Transfer Tutarı", "İşlem Sıklığı", "Hesap Yaşı"]
    explanation = {name: round(score, 4) for name, score in zip(feature_names, importance_scores)}
    
    return explanation