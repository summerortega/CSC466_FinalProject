import sys

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from evaluation import read_csv, print_cluster_report


def main(filename, epsilon, num_points, ground_truth_col=None):
    df, ground_truth = read_csv(filename, ground_truth_col)
    arr = df.to_numpy()

    labels, core_points, neighborhood = dbscan(arr, epsilon, num_points)
    num_outliers = labels.count(-1)
    outlier_percent = 100 * num_outliers / len(labels)

    print("DBSCAN Results")
    print("Number of core points:", len(core_points))
    print("Outlier percentage:", outlier_percent)

    print_cluster_report(arr, labels, ground_truth)

def dbscan(arr, epsilon, num_points):
    n = len(arr)

    matrix = pd.DataFrame(pairwise_distances(arr, metric="euclidean"))

    neighborhood = [np.where(matrix.iloc[i] <= epsilon)[0].tolist()for i in range(n)]

    core_points = [i for i in range(n) if len(neighborhood[i]) >= num_points]

    cluster = [0] * n
    curr_cluster = 0

    for point in core_points:
        if cluster[point] == 0:
            curr_cluster += 1
            cluster[point] = curr_cluster
            density_connected(point, core_points, neighborhood, cluster, curr_cluster)

    labels = []
    for c in cluster:
        if c == 0:
            labels.append(-1)
        else:
            labels.append(c - 1)

    return labels, core_points, neighborhood


def density_connected(point, core_points, neighborhood, cluster, current_cluster):
    for neighbor in neighborhood[point]:
        if cluster[neighbor] == 0:
            cluster[neighbor] = current_cluster

            if neighbor in core_points:
                density_connected(neighbor, core_points, neighborhood, cluster, current_cluster)


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 3:
        main(args[0], float(args[1]), int(args[2]))
    elif len(args) == 4:
        main(args[0], float(args[1]), int(args[2]), int(args[3]))
    else:
        print("Usage: python3 dbscan.py <csv_path> <epsilon> <num_points> [<ground_truth_col>]")