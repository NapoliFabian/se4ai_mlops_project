import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer




def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)



# featurization
def build_features(train_texts, test_texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer


def extract_targets(df):
    return df["label"].values


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# PIPELINE

def featurize(train_path, test_path, output_dir, vectorizer_path):
    # load
    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    # texts
    train_texts = train_df["title"].fillna("")
    test_texts = test_df["title"].fillna("")

    # features
    X_train, X_test, vectorizer = build_features(train_texts, test_texts)

    # labels
    y_train = extract_targets(train_df)
    y_test = extract_targets(test_df)

    # save outputs
    os.makedirs(output_dir, exist_ok=True)

    save_pickle(vectorizer, vectorizer_path)
    save_pickle((X_train, y_train), os.path.join(output_dir, "train.pkl"))
    save_pickle((X_test, y_test), os.path.join(output_dir, "test.pkl"))

    print("Featurization completed.")
    print(f"Train shape: {X_train.shape}")
    print(f"Vectorizer in: {vectorizer_path}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python featurization.py <train_csv> <test_csv> <output_dir> <vectorizer_path>")
        sys.exit(1)

    featurize(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])