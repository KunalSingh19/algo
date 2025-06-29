def rolling_corr(df, col1, col2, window=20):
    return df[col1].rolling(window).corr(df[col2])
