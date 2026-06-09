import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import rand_score

def read_csv(csv_path: str, ground_truth_col=None) -> pd.DataFrame:
    with open(csv_path, "r") as f:
        first_line = f.readline().strip()

    try:
        restrictions = pd.Series(
            [int(value.strip()) for value in first_line.split(",") if value.strip() != ""]
        )
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        df = pd.read_csv(csv_path)
        restrictions = df.iloc[0].astype("int64")
        df = df.drop(index=0).reset_index(drop=True)
        clean_df = restrictions[restrictions == 1].index
        x = df.loc[:, clean_df].astype(float)

        ground_truth = None
        if ground_truth_col is not None:
            ground_truth_col = int(ground_truth_col)
            ground_truth = df.iloc[:, ground_truth_col]

        return x, ground_truth

    num_cols = len(restrictions)
    df = pd.read_csv(csv_path, header=None, usecols=range(num_cols))
    df = df.drop(index=0).reset_index(drop=True)

    clean_df = restrictions[restrictions == 1].index
    x = df.loc[:, clean_df].astype(float)

    ground_truth = None
    if ground_truth_col is not None:
        ground_truth_col = int(ground_truth_col)
        ground_truth = df.iloc[:, ground_truth_col]

    return x, ground_truth

def centroid(points):
    return np.mean(points, axis=0)

def cluster_radius(points):
    c = centroid(points)
    distances = np.linalg.norm(points - c, axis=1)
    return float(np.max(distances))

def intercluster_distance(points1, points2):
    c1 = centroid(points1)
    c2 = centroid(points2)
    return float(np.linalg.norm(c1 - c2))

def can_compute_silhouette(data, labels):
    num_points = len(data)
    num_clusters = len(set(labels))

    return 2 <= num_clusters <= num_points - 1

def cluster_metrics(data, labels, ground_truth=None):
    data = np.array(data)
    labels = np.array(labels)

    unique_labels = sorted(set(labels))

    if can_compute_silhouette(data, labels):
        overall_silhouette = round(float(silhouette_score(data, labels)), 2)
        sample_silhouettes = silhouette_samples(data, labels)
    else:
        overall_silhouette = None
        sample_silhouettes = None

    cluster_points = []
    radii = []
    cluster_rows = {}

    for label in unique_labels:
        points = data[labels == label]
        cluster_points.append((label, points))
        radius = round(cluster_radius(points), 2)
        radii.append(radius)

        cluster_row = {
            "Cluster ID": int(label),
            "Cluster Size": int(len(points)),
            "Radius": radius,
        }

        if sample_silhouettes is not None:
            cluster_silhouettes = sample_silhouettes[labels == label]
            cluster_row["Silhouette Score"] = round(float(np.mean(cluster_silhouettes)), 2)
        else:
            cluster_row["Silhouette Score"] = np.nan

        cluster_rows[int(label)] = cluster_row

    inter_distances = []

    for i in range(len(cluster_points)):
        for j in range(i + 1, len(cluster_points)):
            dist = intercluster_distance(cluster_points[i][1], cluster_points[j][1])
            inter_distances.append(dist)

    if len(inter_distances) > 0 and len(radii) > 0:
        avg_radius = sum(radii) / len(radii)
        avg_intercluster_distance = sum(inter_distances) / len(inter_distances)
        radius_distance_ratio = round(avg_radius / avg_intercluster_distance, 2)
    else:
        radius_distance_ratio = None

    metrics = {
        "overall": {
            "Silhouette Score": overall_silhouette,
            "Rand Index": round(float(rand_score(ground_truth, labels)), 2) if ground_truth is not None else None,
            "Average Radius / Average Intercluster Distance": radius_distance_ratio,
        },
        "clusters": cluster_rows,
    }

    return metrics


def cluster_metrics_table(data, labels, ground_truth=None):
    metrics = cluster_metrics(data, labels, ground_truth)
    return pd.DataFrame(metrics["clusters"].values())

def print_cluster_report(data, labels, ground_truth=None):
    data = np.array(data)
    labels = np.array(labels)
    metrics = cluster_metrics(data, labels, ground_truth)

    print("Overall Metrics")
    overall_metrics = metrics["overall"]

    if overall_metrics["Silhouette Score"] is not None:
        print("Overall silhouette score:", overall_metrics["Silhouette Score"])
    else:
        print("Overall silhouette score: undefined")

    if overall_metrics["Rand Index"] is not None:
        print("Rand Index:", overall_metrics["Rand Index"])

    if overall_metrics["Average Radius / Average Intercluster Distance"] is not None:
        print(
            "Average radius / average intercluster distance:",
            overall_metrics["Average Radius / Average Intercluster Distance"],
        )
    else:
        print("Average radius / average intercluster distance: undefined")