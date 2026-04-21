import os
import numpy as np
import pandas as pd
import pytest

from featurization import (
    load_csv,
    build_tfidf_features,
    build_sbert_features,
    extract_targets,
    save_pickle,
    featurize
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "text": [
            "fake news about politics",
            "real economic update",
            "breaking sports news",
            "fake health claim"
        ],
        "label": [0, 1, 1, 0]
    })
    
    
#test_load
def test_load_csv(tmp_path, sample_df):
    file = tmp_path / "data.csv"
    sample_df.to_csv(file, index=False)

    df = load_csv(file)

    assert len(df) == 4
    assert "text" in df.columns


def test_load_csv_missing():
    with pytest.raises(FileNotFoundError):
        load_csv("non_existing_file.csv")
        

#TFIDF   
def test_build_tfidf_features(sample_df):
    train_texts = sample_df["text"]
    test_texts = sample_df["text"]

    X_train, X_test, vectorizer = build_tfidf_features(train_texts, test_texts)

    # shape consistency
    assert X_train.shape[0] == len(train_texts)
    assert X_test.shape[0] == len(test_texts)

    # sparse matrix
    assert hasattr(X_train, "shape")

    # vocabulary exists
    assert len(vectorizer.vocabulary_) > 0
    
    
def test_extract_targets(sample_df):
    y = extract_targets(sample_df)

    assert len(y) == 4
    assert set(y) == {0, 1}
    
    
def test_save_pickle(tmp_path):
    obj = {"a": 1}
    path = tmp_path / "model.pkl"

    save_pickle(obj, path)

    assert path.exists()
    
def test_featurize_tfidf(tmp_path, sample_df):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"

    sample_df.to_csv(train_file, index=False)
    sample_df.to_csv(test_file, index=False)

    output_dir = tmp_path / "out"
    vectorizer_path = tmp_path / "vec.pkl"

    X_train, X_test, y_train, y_test = featurize(
        train_path=train_file,
        test_path=test_file,
        output_dir=output_dir,
        vectorizer_path=vectorizer_path,
        method="tfidf"
    )

    # output validation
    assert X_train.shape[0] == len(sample_df)
    assert X_test.shape[0] == len(sample_df)

    assert (output_dir / "train.pkl").exists()
    assert (output_dir / "test.pkl").exists()
    assert vectorizer_path.exists()
    
    
def test_featurize_invalid_method(tmp_path, sample_df):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"

    sample_df.to_csv(train_file, index=False)
    sample_df.to_csv(test_file, index=False)

    with pytest.raises(ValueError):
        featurize(
            train_path=train_file,
            test_path=test_file,
            output_dir=tmp_path,
            vectorizer_path=tmp_path / "vec.pkl",
            method="invalid_method"
        )