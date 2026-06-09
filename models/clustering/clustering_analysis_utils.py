import numpy as np
import pandas as pd
from mizani._colors._palettes import brewer
from mizani.formatters import percent_format
from plotnine import *
from sklearn.metrics import rand_score
from sklearn.metrics.pairwise import pairwise_distances

from models.clustering.evaluation import cluster_metrics_table

def cluster_name(cluster_id):
    return f"Cluster {int(cluster_id) + 1}"

def predictions_output(data_path, labels, prediction_output_path, centroids):
    breed_info = (
        pd.read_csv(data_path)
        .drop(index=0)
        .reset_index(drop=True)[["Breed", "AKC Group"]]
        .rename(columns={"AKC Group": "Actual AKC Group"})
    )

    prediction_results = breed_info.copy()
    predicted_groups = cluster_to_akc_predictions(data_path, centroids)
    prediction_results.insert(1, "Trait Cluster", [cluster_name(label) for label in labels])
    prediction_results = prediction_results.merge(predicted_groups, on="Trait Cluster", how="left")
    prediction_results["Correct AKC Prediction"] = (
        prediction_results["Predicted AKC Group"] == prediction_results["Actual AKC Group"]
    )

    prediction_results = prediction_results[
        [
            "Breed",
            "Trait Cluster",
            "Predicted AKC Group",
            "Actual AKC Group",
            "Distance To Predicted AKC Group",
            "Correct AKC Prediction",
        ]
    ]

    return prediction_results


def load_full_trait_data(data_path):
    full_trait_data = pd.read_csv(data_path).drop(index=0).reset_index(drop=True)
    trait_columns = list(full_trait_data.columns[1:15])

    for col in trait_columns:
        full_trait_data[col] = pd.to_numeric(full_trait_data[col], errors="coerce")

    return full_trait_data, trait_columns


def akc_group_traits_table(data_path):
    full_trait_data, trait_columns = load_full_trait_data(data_path)

    group_traits = (
        full_trait_data.groupby("AKC Group")[trait_columns]
        .mean()
        .round(2)
        .T
    )
    group_traits.index.name = "Trait"
    return group_traits


def cluster_to_akc_predictions(data_path, centroids):
    full_trait_data, trait_columns = load_full_trait_data(data_path)
    akc_centroids = full_trait_data.groupby("AKC Group")[trait_columns].mean()

    distance_matrix = pairwise_distances(centroids, akc_centroids.to_numpy(), metric="euclidean")
    nearest_indices = np.argmin(distance_matrix, axis=1)

    rows = []
    akc_group_names = list(akc_centroids.index)

    for cluster_id, nearest_index in enumerate(nearest_indices):
        rows.append(
            {
                "Trait Cluster": cluster_name(cluster_id),
                "Predicted AKC Group": akc_group_names[nearest_index],
                "Distance To Predicted AKC Group": round(float(distance_matrix[cluster_id, nearest_index]), 2),
            }
        )

    return pd.DataFrame(rows)


def cluster_composition_data(prediction_results):
    composition_df = (
        prediction_results.groupby(["Trait Cluster", "Actual AKC Group"])
        .size()
        .reset_index(name="Breed Count")
    )
    sorted_clusters = sorted(
        composition_df["Trait Cluster"].unique(),
        key=lambda label: int(str(label).split()[-1]),
    )
    composition_df["Trait Cluster"] = pd.Categorical(
        composition_df["Trait Cluster"],
        categories=sorted_clusters,
        ordered=True,
    )
    group_totals = (
        composition_df.groupby("Actual AKC Group")["Breed Count"]
        .sum()
        .reset_index()
    )
    sorted_groups = (
        group_totals.sort_values(by="Breed Count", ascending=False)["Actual AKC Group"]
        .tolist()
    )
    composition_df["Actual AKC Group"] = pd.Categorical(
        composition_df["Actual AKC Group"],
        categories=sorted_groups,
        ordered=True,
    )

    return composition_df


def plot_cluster_composition(prediction_results):
    composition_df = cluster_composition_data(prediction_results)

    return (
        ggplot(
            composition_df,
            aes(x="Trait Cluster", y="Breed Count", fill="Actual AKC Group"),
        )
        + geom_col(position=position_fill())
        + scale_fill_discrete(drop=False)
        + scale_y_continuous(labels=percent_format())
        + labs(
            title="AKC Group Composition Within Each Trait Cluster",
            x="Trait Cluster",
            y="Breed Proportion",
            fill="Actual AKC Group",
        )
        + theme_minimal()
        + theme(
            figure_size=(10, 6),
            axis_text_x=element_text(rotation=25, ha="right"),
        )
    )


def cluster_summary_with_akc(prediction_results, data, labels, data_path, centroids):
    labels = np.array(labels)
    metrics_df = cluster_metrics_table(data, labels)
    predicted_groups = cluster_to_akc_predictions(data_path, centroids)

    rows = []

    for cluster_id, cluster in enumerate(sorted(prediction_results["Trait Cluster"].unique())):
        mask = prediction_results["Trait Cluster"] == cluster
        cluster_group = prediction_results.loc[mask]
        most_common_akc = cluster_group["Actual AKC Group"].mode().iloc[0]
        most_common_akc_count = int((cluster_group["Actual AKC Group"] == most_common_akc).sum())
        cluster_member_mask = labels == cluster_id
        akc_member_mask = prediction_results["Actual AKC Group"].to_numpy() == most_common_akc
        metrics_row = metrics_df.loc[metrics_df["Cluster ID"] == cluster_id].iloc[0]
        predicted_row = predicted_groups.loc[predicted_groups["Trait Cluster"] == cluster].iloc[0]

        rows.append(
            {
                "Trait Cluster": cluster,
                "Cluster Size": int(metrics_row["Cluster Size"]),
                "Most Common AKC Group": most_common_akc,
                "Most Common AKC Group Count": most_common_akc_count,
                "Predicted AKC Group": predicted_row["Predicted AKC Group"],
                "Distance To Predicted AKC Group": predicted_row["Distance To Predicted AKC Group"],
                "AKC Rand Index": round(float(rand_score(cluster_member_mask, akc_member_mask)), 2),
                "Silhouette Score": metrics_row["Silhouette Score"],
                "Radius": metrics_row["Radius"],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.set_index("Trait Cluster").T
    summary_df.index.name = "Metric"

    return summary_df
