import json
import os
from pathlib import Path
import pickle
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification, BertTokenizer

from models.bert_classifier import predict_bert

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
# LOAD

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_test_data(test_path):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not founded: {test_path}")

    with open(test_path, "rb") as f:
        return pickle.load(f)



def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }



def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


# PIPELINE
#

def evaluate(model_type, model_path, test_data, metrics_out):

    if model_type == "bert":

        # load model
        model = BertForSequenceClassification.from_pretrained("models/")
        tokenizer = BertTokenizer.from_pretrained("models/")

        # load CSV (NON pickle)
        df = pd.read_csv("data/interim/test.csv")
        texts = df["title"].fillna("").tolist()
        y_test = df["label"].values

        preds = predict_bert(model, tokenizer, texts)

    else:

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(test_data, "rb") as f:
            X_test, y_test = pickle.load(f)

        preds = model.predict(X_test)
    # metrics
    metrics = compute_metrics(y_test, preds)

    # save
    save_metrics(metrics, metrics_out)

    # log
    print(f"Accuracy : {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall   : {metrics['recall']:.2f}")
    print(f"F1 Score : {metrics['f1_score']:.2f}")

    print(f"Metrics in {metrics_out}")

    return metrics



if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python evaluate.py <model_type> <model_path> <test_data> <metrics_out>")
        sys.exit(1)

    model_type = sys.argv[1]
    model_path = sys.argv[2]
    test_data = sys.argv[3]
    metrics_out = sys.argv[4]

    evaluate(model_type, model_path, test_data, metrics_out)