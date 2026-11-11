import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import mlflow
import mlflow.pytorch

class BankFraudGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(BankFraudGAT, self).__init__()
        # Graph Attention Network (GAT) katmanları
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train():
    data = torch.load("banking_graph.pt", weights_only=False)
    
    # MLflow Deney Başlatma
    mlflow.set_experiment("Bank_Fraud_GAT_Detection")
    
    with mlflow.start_run():
        lr = 0.01
        hidden_dim = 16
        epochs = 100
        
        # Parametreleri MLflow'a kaydetme
        mlflow.log_param("architecture", "GAT")
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("epochs", epochs)

        model = BankFraudGAT(in_channels=3, hidden_channels=hidden_dim, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            
            # Metriği her 10 epoch'ta bir kaydetme
            if epoch % 10 == 0:
                mlflow.log_metric("loss", loss.item(), step=epoch)
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

        # Model Ağırlıklarını Saklama
        torch.save(model.state_dict(), "gnn_fraud_model.pt")
        mlflow.pytorch.log_model(model, "gat_fraud_model")
        
        print("GAT Modeli eğitimi tamamlandı, MLflow'a kaydedildi ve 'gnn_fraud_model.pt' üretildi.")

if __name__ == "__main__":
    train()