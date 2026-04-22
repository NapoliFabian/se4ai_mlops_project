import os
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# LOAD

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


# TF-IDF

def build_tfidf_features(train_texts, test_texts, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer


# SENTENCE BERT

def build_sbert_features(train_texts, test_texts, model_name="all-MiniLM-L6-v2"):
    """
    Sentence-BERT embeddings (mean pooling già gestito dal modello)
    """
    model = SentenceTransformer(model_name)

    X_train = model.encode(train_texts.tolist(), show_progress_bar=True)
    X_test = model.encode(test_texts.tolist(), show_progress_bar=True)

    return np.array(X_train), np.array(X_test), model



def extract_targets(df):
    return df["label"].values

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# PIPELINE

def featurize(
    train_path,
    test_path,
    output_dir,
    vectorizer_path,
    method="tfidf" #choise
):
    # LOAD
    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    train_texts = train_df["text"].fillna("")
    test_texts = test_df["text"].fillna("")

    y_train = extract_targets(train_df)
    y_test = extract_targets(test_df)

    # METHOD
    if method.lower() == "tfidf":
        print(">>> Using TF-IDF")
        X_train, X_test, encoder = build_tfidf_features(train_texts, test_texts)

    elif method.lower() == "sbert":
        print(">>> Using Sentence-BERT embeddings")
        X_train, X_test, encoder = build_sbert_features(train_texts, test_texts)

    else:
        raise ValueError(f"Unknown method: {method}")

    # SAVE
    os.makedirs(output_dir, exist_ok=True)

    save_pickle(encoder, vectorizer_path)
    save_pickle((X_train, y_train), os.path.join(output_dir, "train.pkl"))
    save_pickle((X_test, y_test), os.path.join(output_dir, "test.pkl"))

    print("Featurization completed.")
    print(f"Method: {method}")
    print(f"Train shape: {X_train.shape}")

    return X_train, X_test, y_train, y_test


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6:
        print("Usage: python featurization.py <train_csv> <test_csv> <output_dir> <vectorizer_path> <method>")
        sys.exit(1)

    featurize(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5]
    )