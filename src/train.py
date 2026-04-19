import pickle
import sys
import os
from sklearn.linear_model import LogisticRegression


def load_train_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)



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



def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


# PIPELINE

def train_model(train_input, model_out, seed):
    # load
    X_train, y_train = load_train_data(train_input)

    # train
    model = train_logistic_regression(X_train, y_train, seed)

    # save
    save_model(model, model_out)
    print(f"Logistic Regression in {model_out}")

    return model


# =========================
# CLI
# =========================

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_model.py <train_input> <model_out> <seed>")
        sys.exit(1)

    train_model(sys.argv[1], sys.argv[2], sys.argv[3])