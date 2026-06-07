from pathlib import Path

import numpy as np
import pandas as pd


TRAIT_GROUPS = {
    "Family Life": [
        "Affectionate With Family",
        "Good With Young Children",
        "Good With Other Dogs",
    ],
    "Physical": [
        "Shedding Level",
        "Coat Grooming Frequency",
        "Drooling Level",
    ],
    "Social": [
        "Openness To Strangers",
        "Playfulness Level",
        "Watchdog/Protective Nature",
        "Adaptability Level",
    ],
    "Personality": [
        "Trainability Level",
        "Energy Level",
        "Barking Level",
        "Mental Stimulation Needs",
    ],
}

CATEGORICAL_TRAIT_COLS = ["Coat Type", "Coat Length"]


def find_project_root(start=None):
    start = Path.cwd() if start is None else Path(start)

    for base_dir in [start, *start.parents]:
        if (base_dir / "data").exists():
            return base_dir

    raise FileNotFoundError("Could not find project root with a data/ folder.")


def data_dirs(project_root):
    interim_dir = project_root / "data" / "interim"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return interim_dir, processed_dir


def numeric_trait_cols():
    return [col for cols in TRAIT_GROUPS.values() for col in cols]


def load_interim_data(interim_dir, include_groups=False):
    breed_traits = pd.read_csv(interim_dir / "breed_traits.csv")
    breed_ranks = pd.read_csv(interim_dir / "breed_ranks.csv")

    if include_groups:
        breed_groups = pd.read_csv(interim_dir / "breed_groups.csv")
        return breed_traits, breed_ranks, breed_groups

    return breed_traits, breed_ranks


def add_popularity_tier(breed_ranks):
    breed_ranks = breed_ranks.copy()
    rank_cols = [f"{year} Rank" for year in range(2013, 2026)]
    average_rank = np.nanmean(breed_ranks[rank_cols].to_numpy(dtype=float), axis=1)

    breed_ranks["Average Rank"] = average_rank
    breed_ranks["Popularity Tier"] = pd.cut(
        average_rank,
        bins=[0, 50, 125, np.inf],
        labels=["High Popularity", "Medium Popularity", "Low Popularity"],
        include_lowest=True,
    )

    return breed_ranks


def prepare_numeric_traits(breed_traits):
    breed_traits = breed_traits.copy()
    cols = numeric_trait_cols()
    breed_traits[cols] = breed_traits[cols].apply(pd.to_numeric, errors="coerce")
    return breed_traits, cols


def build_avg_traits(breed_traits):
    avg_traits = breed_traits[["Breed"]].copy()
    avg_cols = []

    for group_name, cols in TRAIT_GROUPS.items():
        avg_traits[group_name] = breed_traits[cols].mean(axis=1)
        avg_cols.append(group_name)

    return avg_traits, avg_cols


def add_random_forest_metadata(model_features, numeric_cols, class_col="Popularity Tier"):
    domain_codes = []

    for col in model_features.columns:
        if col == "Breed":
            domain_codes.append(-1)
        elif col in numeric_cols:
            domain_codes.append(0)
        else:
            domain_codes.append(model_features[col].nunique())

    domain_code_row = pd.DataFrame([domain_codes], columns=model_features.columns)
    class_variable_row = pd.DataFrame(
        [[class_col] + [""] * (len(model_features.columns) - 1)],
        columns=model_features.columns,
    )

    return pd.concat([domain_code_row, class_variable_row, model_features], ignore_index=True)


def add_clustering_metadata(model_features, clustering_cols):
    metadata_codes = [1 if col in clustering_cols else 0 for col in model_features.columns]
    metadata_row = pd.DataFrame([metadata_codes], columns=model_features.columns, dtype=object)
    return pd.concat([metadata_row, model_features.astype(object)], ignore_index=True)


def write_random_forest_csv(model_features, numeric_cols, output_path):
    model_features_with_metadata = add_random_forest_metadata(model_features, numeric_cols)
    model_features_with_metadata.to_csv(output_path, index=False)
    return output_path, model_features_with_metadata


def write_clustering_csv(model_features, clustering_cols, output_path):
    model_features_with_metadata = add_clustering_metadata(model_features, clustering_cols)
    model_features_with_metadata.to_csv(output_path, index=False)
    return output_path, model_features_with_metadata
