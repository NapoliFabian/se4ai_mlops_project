import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DenseClassifier, self).__init__()
        # Definizione dell'architettura
        # input_dim sarà il numero di feature generate da TfidfVectorizer
        self.fc1 = nn.Linear(input_dim, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # Output singolo per classificazione binaria
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Passaggio attraverso i layer con attivazione ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def predict(self, X):
            self.eval() # Imposta in modalità valutazione
            with torch.no_grad():
                # Converte in tensore se è un array/matrice sparsa
                if hasattr(X, "toarray"): X = X.toarray()
                inputs = torch.FloatTensor(X)
                
                outputs = self.forward(inputs)
                # Converte le probabilità (0-1) in classi (0 o 1)
                return (outputs > 0.5).int().numpy().flatten()