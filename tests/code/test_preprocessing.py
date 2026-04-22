import pandas as pd
import pytest
from io import StringIO

from preprocessing import (
    load_data,
    clean_title,
    drop_empty_text,
    clean_text,
    apply_text_cleaning,
    preprocess_dataset,
)


# FIXTURES

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "title": [None, "News title", "Another"],
        "text": ["Hello WORLD!!!", "", None]
    })


@pytest.fixture
def sample_csv(tmp_path):
    csv_content = """title,text
    ,Hello WORLD!!!
    News title,
    Another,Valid text!!!
    """
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_content)
    return file_path



def test_load_data(sample_csv):
    df = load_data(sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_clean_title(sample_df):
    df = clean_title(sample_df)

    assert df["title"].isna().sum() == 0
    assert df["title"].iloc[0] == ""



def test_drop_empty_text(sample_df):
    df = drop_empty_text(sample_df)

    assert len(df) == 1
    assert df["text"].iloc[0] == "Hello WORLD!!!"



@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("Hello WORLD!!#@!", "hello world"),
        ("Test   multiple   spaces", "test multiple spaces"),
        ("Special #$%^ chars", "special chars"),
        ("123 NUMBERS", "123 numbers"),
    ],
)
def test_clean_text(input_text, expected):
    assert clean_text(input_text) == expected



def test_apply_text_cleaning():
    df = pd.DataFrame({
        "text": ["Hello!!#@!", "Another   TEXT"]
    })

    result = apply_text_cleaning(df)

    assert result["text"].iloc[0] == "hello"
    assert result["text"].iloc[1] == "another text"




def test_preprocess_dataset(sample_csv):
    df = preprocess_dataset(sample_csv, output_path=None)

    assert len(df) == 2
    assert df["title"].isna().sum() == 0

    assert all(df["text"].str.contains(r"[^a-z0-9\s]") == False)