import numpy as np
import pandas as pd

def train_test_split(df, test_size=0.2):
    n = len(df)
    test_len = int(n * test_size)
    train_df = df.iloc[:-test_len]
    test_df = df.iloc[-test_len:]
    return train_df, test_df

def walk_forward_splits(df, window_train, window_test, step=None):
    if step is None:
        step = window_test
    splits = []
    for start in range(0, len(df) - window_train - window_test, step):
        splits.append((start, start + window_train))
    return splits
