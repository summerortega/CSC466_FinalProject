import pandas as pd

#helper function that assists
#in reading csv, properly typing
#all columns, and identifying class
#attribute
def read_csv(csv_path:str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # read entire csv
    df = pd.read_csv(csv_path)

    #parsing first 3 rows of df
    col_names = pd.Series(df.columns, index=df.columns)
    data_types = df.iloc[0].astype("int64")
    class_var = df.iloc[1, 0]

    rowid_cols = col_names[data_types == -1]
    a = col_names[(col_names != class_var) & (~col_names.isin(rowid_cols))]

    # drop metadata
    df = df.drop([0, 1]).reset_index(drop=True)

    col_types = pd.Series(
        ["float64" if t == 0 else "category" for t in data_types if t != -1],
        index=col_names[data_types != -1]
    ).to_dict()

    df = df.astype(col_types)

    #separate to y and X
    y = df.loc[:, class_var]
    x = df.drop(columns=[class_var, *rowid_cols.tolist()], errors="ignore")
    return x, y, a.reset_index(drop=True)