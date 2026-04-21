import os
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# split function
def split_dataframe(df, test_size, seed, target_column="label"):
    if target_column not in df.columns:
        raise ValueError(f"Target Column '{target_column}' not found")

    train_df, test_df = train_test_split(
        df,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=df[target_column]
    )

    return train_df, test_df



def load_csv(input_path):
    print(f"Loading{input_path}" )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not founded: {input_path}")

    return pd.read_csv(input_path)

def save_csv(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


# pipeline

def split_data(input_path, output_train, output_test, test_size, seed):
    # load
    df = load_csv(input_path)

    # split
    train_df, test_df = split_dataframe(df, test_size, seed)
    print(f"Split: {len(train_df)} train / {len(test_df)} test")
    # save
    save_csv(train_df, output_train)
    save_csv(test_df, output_test)

    

    return train_df, test_df


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python split_dataset.py <input_path> <test_size> <seed> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    test_size = sys.argv[2]
    seed = sys.argv[3]
    output_dir = sys.argv[4]

    output_train = os.path.join(output_dir, "train.csv")
    output_test = os.path.join(output_dir, "test.csv")

    split_data(input_path, output_train, output_test, test_size, seed)