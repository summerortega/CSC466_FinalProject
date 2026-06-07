import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from models.randomForest.randomForest import RandomForest
from models.readCSV import read_csv


def build_rf(params, seed=42):
    return RandomForest(
        num_attributes=int(params["num_attributes"]),
        num_datapoints=float(params["num_datapoints"]),
        num_trees=int(params["num_trees"]),
        splitting_threshold=float(params["splitting_threshold"]),
        random_state=seed,
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        balanced_bootstrap=(params["balanced_bootstrap"]),
    )

def score_params_cv(params, x_train, y_train, attributes, cv_splits=5, seed=42):
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    macro_f1_scores = []

    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(x_train, y_train), start=1):
        rf = build_rf(params, seed=seed + fold_number)
        rf.fit(x_train.iloc[train_idx], y_train.iloc[train_idx], attributes)
        predictions = rf.predict(x_train.iloc[val_idx])
        y_val = y_train.iloc[val_idx]

        macro_f1_scores.append(f1_score(y_val, predictions, average="macro", zero_division=0))

    return float(np.mean(macro_f1_scores))


def run_grid_search(data_path, grid_path):
    x, y, attributes = read_csv(str(data_path))
    grid = pd.read_csv(str(grid_path))

    x_train, _, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []

    for i, params_row in grid.iterrows():
        params = params_row.to_dict()
        score = score_params_cv(params, x_train, y_train, attributes)
        results.append({**params, "mean_cv_macro_f1": score})

    results_df = pd.DataFrame(results).sort_values("mean_cv_macro_f1", ascending=False)
    best_params = results_df.iloc[0].drop("mean_cv_macro_f1").to_dict()
    return best_params
