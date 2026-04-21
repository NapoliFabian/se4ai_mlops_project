import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DenseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def predict(self, X):
            self.eval()
            with torch.no_grad():
                if hasattr(X, "toarray"): X = X.toarray()
                inputs = torch.FloatTensor(X)
                
                outputs = self.forward(inputs)
                return (outputs > 0.5).int().numpy().flatten()