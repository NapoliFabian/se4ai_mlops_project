import os
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.neural_network import DenseClassifier

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

def evaluate(model_path, test_pkl, metrics_out):
    # load
    model = load_model(model_path)
    X_test, y_test = load_test_data(test_pkl)

    # predict
    y_pred = model.predict(X_test)

    # metrics
    metrics = compute_metrics(y_test, y_pred)

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
    import sys

    if len(sys.argv) != 4:
        print("Usage: python evaluate.py <model_path> <test_pkl> <metrics_out>")
        sys.exit(1)

    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])