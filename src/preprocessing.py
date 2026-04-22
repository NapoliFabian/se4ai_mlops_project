import logging
import re
import pandas as pd


# LOGGING CONFIG
def setup_logger():
    logger = logging.getLogger("data_cleaning")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger



def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_title(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("")
    return df


def drop_empty_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]

    return df


def clean_text(text: str) -> str:
    """
    Rimuove caratteri speciali, mantiene solo contenuto testuale
    """
    text = text.lower()

    # rimuove caratteri non alfanumerici
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # rimuove spazi multipli
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    return df

def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    logger = setup_logger()

    logger.info(f"Loading dataset from: {input_path}")
    df = load_data(input_path)

    logger.info(f"Initial shape: {df.shape}")

    # CLEAN TITLE
    null_titles = df["title"].isna().sum()
    logger.info(f"Null titles found: {null_titles}")
    df = clean_title(df)

    # DROP EMPTY TEXT
    before_rows = len(df)
    df = drop_empty_text(df)
    after_rows = len(df)

    logger.info(f"Dropped {before_rows - after_rows} rows with empty/null text")

    # CLEAN TEXT
    logger.info("Cleaning text (removing special characters)")
    df = apply_text_cleaning(df)

    logger.info(f"Final shape: {df.shape}")

    # SAVE
    #df.to_csv(output_path, index=False)
    #logger.info(f"Saved cleaned dataset to: {output_path}")

    return df


# ---------------------------
# CLI ENTRY
# ---------------------------
if __name__ == "__main__":
    preprocess_dataset(
        input_path="data/raw/dataset.csv",
        output_path="data/processed/cleaned_dataset.csv"
    )