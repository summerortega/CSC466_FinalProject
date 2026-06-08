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

def print_cluster_report(data, labels, ground_truth=None):
    data = np.array(data)
    labels = np.array(labels)

    non_noise_mask = labels != -1
    clean_data = data[non_noise_mask]
    clean_labels = labels[non_noise_mask]

    unique_labels = sorted(set(labels))
    clean_unique_labels = sorted(set(clean_labels))

    print("\nOverall Metrics")

    if can_compute_silhouette(clean_data, clean_labels):
        overall_silhouette = silhouette_score(clean_data, clean_labels)
        sample_silhouettes = silhouette_samples(clean_data, clean_labels)
        print("Overall silhouette score:", overall_silhouette)
    else:
        sample_silhouettes = None
        print("Overall silhouette score: undefined")

    if ground_truth is not None:
        print("Rand Index:", rand_score(ground_truth, labels))

    cluster_points = []
    radii = []

    for label in clean_unique_labels:
        points = data[labels == label]
        cluster_points.append((label, points))
        radii.append(cluster_radius(points))

    inter_distances = []

    for i in range(len(cluster_points)):
        for j in range(i + 1, len(cluster_points)):
            dist = intercluster_distance(cluster_points[i][1], cluster_points[j][1])
            inter_distances.append(dist)

    if len(inter_distances) > 0:
        avg_radius = sum(radii) / len(radii)
        avg_intercluster_distance = sum(inter_distances) / len(inter_distances)
        ratio = avg_radius / avg_intercluster_distance
        print("Average radius / average intercluster distance:", ratio)
    else:
        print("Average radius / average intercluster distance: undefined")

    print("\nCluster Descriptions")

    for label in unique_labels:
        points = data[labels == label]

        if label == -1:
            print("\nNumber of outliers:", len(points))
            if len(points) <= 10:
                print("Outliers:")
                for point in points:
                    print(point.tolist())
            else:
                print("Representative outliers:")
                for point in points[:5]:
                    print(point.tolist())
            continue

        print(f"\nCluster {label + 1}:")
        print("Number of points:", len(points))

        if len(points) <= 10:
            print("Points:")
            for point in points:
                print(point.tolist())
        else:
            print("Representative points:")
            for point in points[:5]:
                print(point.tolist())

        c = centroid(points)
        r = cluster_radius(points)

        print("Centroid:", c.tolist())
        print("Radius:", r)

        if sample_silhouettes is not None:
            cluster_silhouettes = sample_silhouettes[clean_labels == label]
            print("Cluster silhouette score:", float(np.mean(cluster_silhouettes)))
        else:
            print("Cluster silhouette score: undefined")
