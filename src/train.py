import os
import pickle
import sys

from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

# BERT
from models.bert_classifier import train_bert_classifier
from models.neural_network import DenseClassifier


def load_train_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


# LOGISTIC REGRESSION

def build_logistic_model(seed):
    return LogisticRegression(
        random_state=int(seed),
        max_iter=1000,
        solver="liblinear"
    )


def train_logistic_regression(X_train, y_train, seed):
    model = build_logistic_model(seed)
    model.fit(X_train, y_train)
    return model


# NEURAL NETWORK 

def train_dense_model(X_train, y_train, input_dim, epochs=50, lr=0.002):

    model = DenseClassifier(input_dim)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)

    model.train()

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"[Epoch {epoch+1}] Loss: {loss.item():.6f}")

    return model, loss_history


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)


# PIPELINE

def train_model(train_input, model_out, seed, model_type="logreg"):

    X_train, y_train = load_train_data(train_input)

    # LOGISTIC REGRESSION
    if model_type.lower() == "logreg":

        model = train_logistic_regression(X_train, y_train, seed)
        save_model(model, model_out)

        print(f"[LOGREG] saved in {model_out}")
        return model, []


    # NEURAL NETWORK (SBERT)
    
    elif model_type.lower() == "nn":

        input_dim = X_train.shape[1]

        model, loss_history = train_dense_model(
            X_train,
            y_train,
            input_dim=input_dim
        )

        save_model(model, model_out)

        print(f"[NN] saved in {model_out}")
        return model, loss_history


    # BERT
    elif model_type.lower() == "bert":

        model, tokenizer = train_bert_classifier(
                train_csv="data/interim/train.csv",  
                test_csv="data/interim/test.csv"
            )

        print("[BERT] training completed")
        return model, tokenizer


    else:
        raise ValueError(f"Unknown model_type: {model_type}")




if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python train.py <train_input> <model_out> <seed> <model_type>")
        sys.exit(1)

    train_model(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4]
    )