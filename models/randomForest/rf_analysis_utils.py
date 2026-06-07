import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from models.readCSV import read_csv
from models.setup_utils import add_popularity_tier


LABEL_ORDER = ["High Popularity", "Medium Popularity", "Low Popularity"]


def load_train_test_data(data_path, test_size=0.20, random_state=42):
    x, y, attributes = read_csv(str(data_path))
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return x, y, attributes, x_train, x_test, y_train, y_test


def collect_split_counts(forest):
    split_counts = {}

    def walk(tree_part):
        if "node" not in tree_part:
            return

        node = tree_part["node"]
        feature = node["var"]
        split_counts[feature] = split_counts.get(feature, 0) + 1

        for edge_wrapper in node["edges"]:
            edge = edge_wrapper["edge"]
            if "node" in edge:
                walk({"node": edge["node"]})

    for tree in forest:
        walk(tree.tree)

    return (
        pd.DataFrame(
            [{"Feature": feature, "Split Count": count} for feature, count in split_counts.items()]
        )
        .sort_values("Split Count", ascending=False)
        .reset_index(drop=True)
    )


def feature_profiles(x, y, label_order=LABEL_ORDER):
    numeric_cols = [col for col in x.columns if x[col].dtype.name != "category"]
    categorical_cols = [col for col in x.columns if x[col].dtype.name == "category"]

    numeric_profile = (
        x.assign(Popularity_Tier=y.astype(str))
        .groupby("Popularity_Tier")[numeric_cols]
        .mean()
        .reindex(label_order)
        .round(2)
        .T
    )

    if categorical_cols:
        categorical_profile = (
            x.assign(Popularity_Tier=y.astype(str))
            .groupby("Popularity_Tier")[categorical_cols]
            .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "")
            .reindex(label_order)
        )
    else:
        categorical_profile = pd.DataFrame(index=label_order)

    return numeric_profile, categorical_profile


def make_prediction_results(data_path, ranks_path, x_test, y_test, predictions, rank_decimals=1):
    breed_names = (
        pd.read_csv(data_path)
        .drop(index=[0, 1])
        .reset_index(drop=True)["Breed"]
    )
    rank_lookup = add_popularity_tier(pd.read_csv(ranks_path))[["Breed", "Average Rank"]]
    test_breeds = breed_names.loc[x_test.index].reset_index(drop=True)

    prediction_results = pd.DataFrame(
        {
            "Breed": test_breeds,
            "Actual Popularity Tier": y_test.astype(str).reset_index(drop=True),
            "Predicted Popularity Tier": pd.Series(predictions).astype(str),
        }
    ).merge(rank_lookup, on="Breed", how="left")

    prediction_results["Average Rank"] = prediction_results["Average Rank"].round(rank_decimals)
    prediction_results["Correct"] = (
        prediction_results["Actual Popularity Tier"]
        == prediction_results["Predicted Popularity Tier"]
    )

    return prediction_results[
        [
            "Breed",
            "Average Rank",
            "Actual Popularity Tier",
            "Predicted Popularity Tier",
            "Correct",
        ]
    ]

def plot_confusion(y_test, predictions, title, label_order=LABEL_ORDER, cmap="Blues"):
    cm = confusion_matrix(y_test, predictions, labels=label_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
    disp.plot(xticks_rotation=45, cmap=cmap)
    disp.ax_.set_title(title)
    disp.figure_.tight_layout()
    return disp
