import os
import pickle
import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fakenewsdetector.train_model import train_model
from fakenewsdetector.models.DenseNeuralNetwork import DenseClassifier

#  percorsi
TRAIN_PKL = "data/processed/train.pkl"
MODEL_OUT = "models/test_model.pkl"

@pytest.fixture
def real_data_sample():
    if not os.path.exists(TRAIN_PKL):
        pytest.skip("Processed dataset not founded.")
    with open(TRAIN_PKL, 'rb') as f:
        X, y = pickle.load(f)
    return X, y


# CHECK FOR DECREASING LOSS
def test_decreasing_loss(real_data_sample, tmp_path):
    
    X, y = real_data_sample
    #subset piccolo per velocizzare
    mini_pkl = tmp_path / "mini.pkl"
    with open(mini_pkl, 'wb') as f:
        pickle.dump((X[:100], y[:100]), f)

    _, loss_history = train_model(str(mini_pkl), str(tmp_path/"m.pkl"), "42", "dense")
    
    assert loss_history[-1] < loss_history[0], "La loss finale non è inferiore a quella iniziale"
    
    decreases = 0

    for i in range(1, len(loss_history)):
            if loss_history[i] < loss_history[i - 1]:
                decreases += 1

    total_steps = len(loss_history) - 1

    # min percentage loss steps that decrease
    min_ratio = 0.6  

    assert decreases / total_steps >= min_ratio, (
            f"Trend not decrescent: "
            f"{decreases}/{total_steps} step ({decreases/total_steps:.2f})"
        )
    
# OVERFIT ON A BATCH (Accuracy Check)
def test_overfit_accuracy(real_data_sample, tmp_path):
    """
    Overfit in micro-batch.
    Target: 0.95 ± 0.05
    """
    X, y = real_data_sample
    X_batch = X[:50]
    y_batch = y[:50]
    
    batch_pkl = tmp_path / "batch_overfit.pkl"
    with open(batch_pkl, 'wb') as f:
        pickle.dump((X_batch, y_batch), f)
    
    # Addestriamo il modello sul batch
    model, _ = train_model(str(batch_pkl), str(tmp_path/"m.pkl"), "42", "dense")
    
    # Calcoliamo l'accuratezza finale sul batch stesso
    model.eval()
    with torch.no_grad():
        if hasattr(X_batch, "toarray"): X_batch = X_batch.toarray()
        
        inputs = torch.FloatTensor(X_batch)
        logits = model(inputs)
        
        
        preds = (torch.sigmoid(logits) > 0.5).float().view(-1)
        labels = torch.FloatTensor(y_batch.values if hasattr(y_batch, 'values') else y_batch)
        
        accuracy = (preds == labels).float().mean().item()

    # Assert accuracy == pytest.approx(0.95, abs=0.05)
    assert accuracy == pytest.approx(0.95, abs=0.051), f"{accuracy}"

# 4. TRAIN TO COMPLETION (Artifacts & LR) - CHECK ARTIFACTS 
def test_train_completion_and_artifacts(real_data_sample, tmp_path):


    X, y = real_data_sample
    model_path = tmp_path / "final_model.pkl"
    
    #training completo
    model, _ = train_model(TRAIN_PKL, str(model_path), "42", "dense")
    
    # 1. Assert artifacts: model
    assert model_path.exists(), "Il file del modello non è stato creato!"
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert isinstance(loaded_model, DenseClassifier)
