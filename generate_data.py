import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def generate_synthetic_banking_graph(num_nodes=500, num_edges=2000):
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Düğüm Özellikleri (Node Features): [Ort. Transfer Tutarı, İşlem Sıklığı, Hesap Yaşı (Gün)]
    x = torch.tensor(np.random.randn(num_nodes, 3), dtype=torch.float)
    
    # Etiketler (Labels): 0 = Normal Müşteri, 1 = Fraud / Şüpheli Müşteri (~%10 fraud)
    y = torch.tensor(np.random.choice([0, 1], size=num_nodes, p=[0.9, 0.1]), dtype=torch.long)
    
    # Kenarlar (Edges): Müşteriler arası para transferleri (Source -> Target)
    edge_src = np.random.randint(0, num_nodes, size=num_edges)
    edge_dst = np.random.randint(0, num_nodes, size=num_edges)
    edge_index = torch.tensor(np.array([edge_src, edge_dst]), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    torch.save(data, "banking_graph.pt")
    print(f"Graph verisi oluşturuldu: {num_nodes} Düğüm, {num_edges} Transfer Kenarı.")

if __name__ == "__main__":
    generate_synthetic_banking_graph()