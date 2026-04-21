import pandas as pd
import pytest
from pathlib import Path


from split_dataset import (
    split_dataframe,
    load_csv,
    save_csv,
    split_data
)


# Fixtures

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "title": [f"title_{i}" for i in range(10)],
        "text": [f"text_{i}" for i in range(10)],
        "label": [0, 1] * 5
    })


@pytest.fixture
def tmp_csv(tmp_path, sample_df):
    file_path = tmp_path / "data.csv"
    sample_df.to_csv(file_path, index=False)
    return file_path


def test_split_dataframe_row_columns(sample_df):
    train_df, test_df = split_dataframe(sample_df, test_size=0.2, seed=42)

    assert len(train_df) == 8
    assert len(test_df) == 2
    assert list(train_df.columns) == list(sample_df.columns)


def test_split_dataframe_stratification(sample_df):
    train_df, test_df = split_dataframe(sample_df, test_size=0.2, seed=42)

    original_counts = sample_df["label"].value_counts()
    train_counts = train_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()

    # proporzione mantenuta (approssimata sui conteggi)
    for label in original_counts.index:
        expected_train = original_counts[label] * 0.8
        expected_test = original_counts[label] * 0.2

        assert train_counts[label] == pytest.approx(expected_train, abs=1)
        assert test_counts[label] == pytest.approx(expected_test, abs=1)



def test_split_dataframe_missing_target(sample_df):
    df = sample_df.drop(columns=["label"])

    with pytest.raises(ValueError):
        split_dataframe(df, test_size=0.2, seed=42)


def test_split_dataframe_deterministic(sample_df):
    train1, test1 = split_dataframe(sample_df, 0.2, 42)
    train2, test2 = split_dataframe(sample_df, 0.2, 42)

    pd.testing.assert_frame_equal(train1.reset_index(drop=True),
                                  train2.reset_index(drop=True))
    pd.testing.assert_frame_equal(test1.reset_index(drop=True),
                                  test2.reset_index(drop=True))


def test_load_csv_success(tmp_csv):
    df = load_csv(tmp_csv)

    assert not df.empty
    assert "label" in df.columns
    assert "text" in df.columns
    assert "title" in df.columns

def test_load_csv_file_not_found(tmp_path):
    fake_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(fake_path)


def test_save_csv_creates_file(tmp_path, sample_df):
    output_path = tmp_path / "subdir" / "output.csv"

    save_csv(sample_df, output_path)

    assert output_path.exists()

    loaded = pd.read_csv(output_path)
    assert len(loaded) == len(sample_df)



def test_split_data_complete(tmp_csv, tmp_path):
    output_train = tmp_path / "train.csv"
    output_test = tmp_path / "test.csv"

    train_df, test_df = split_data(
        input_path=tmp_csv,
        output_train=output_train,
        output_test=output_test,
        test_size=0.2,
        seed=42
    )

    # check files
    assert output_train.exists()
    assert output_test.exists()

    # check sizes
    assert len(train_df) == 8
    assert len(test_df) == 2

    # reload and verify
    train_loaded = pd.read_csv(output_train)
    test_loaded = pd.read_csv(output_test)

    #check len
    assert len(train_loaded) == len(train_df)
    assert len(test_loaded) == len(test_df)