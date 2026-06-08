import json

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from .evaluation import read_csv, print_cluster_report
def hclustering_main(filename, threshold=None, ground_truth_col=None):
    df, ground_truth = read_csv(filename, ground_truth_col)
    arr = df.to_numpy()
    data_points = [x.tolist() for x in arr]

    clusters = [
        {"type": "leaf", "data": x.tolist()}
        for x in arr
    ]
    matrix = pd.DataFrame(pairwise_distances(data_points, Y=None, metric='euclidean'))

    while len(matrix) > 1:
        temp_values = matrix.to_numpy(copy=True)
        np.fill_diagonal(temp_values, np.inf)
        temp_matrix = pd.DataFrame(temp_values, index=matrix.index, columns=matrix.columns)

        a = temp_matrix.min(axis=0).idxmin()
        b = temp_matrix[a].idxmin()

        merge_height = float(matrix.iloc[a, b])

        row_a = matrix.iloc[a]
        row_b = matrix.iloc[b]

        min_arr = np.minimum(row_a, row_b)

        matrix.iloc[a] = min_arr
        matrix.iloc[:, a] = min_arr
        matrix.iloc[a, a] = 0

        clusters[a] = {
            "type": "node",
            "height": merge_height,
            "nodes": [clusters[a], clusters[b]]
        }

        matrix = matrix.drop(index=b, columns = b)
        clusters.pop(b)

        matrix = matrix.reset_index(drop=True)
        matrix.columns = range(matrix.columns.size)

    dendrogram = clusters[0]
    dendrogram['type'] = 'root'
    print(json.dumps(dendrogram, indent=2))
    with open("dendrogram.json", "w") as f:
        json.dump(dendrogram, f, indent=2)

    if threshold is not None:
        cut_clusters = cut_tree(dendrogram, threshold)

        labels = labels_from_cut_clusters(arr, cut_clusters)

        print_cluster_report(arr, labels, ground_truth)

        with open("clusters_cut.txt", "w") as f:
            f.write(f"Clusters after cutting at threshold {threshold}\n\n")

            for i, cluster in enumerate(cut_clusters, start=1):
                f.write(f"Cluster {i}:\n")
                for point in cluster:
                    f.write(json.dumps(point, separators=(",", ": ")) + "\n")
                f.write("\n")

        with open("cut_clusters.json", "w") as f:
            json.dump(cut_clusters, f, indent=2)

def get_leafs(tree):
    if tree["type"] == "leaf":
        return [tree["data"]]

    points = []
    for child in tree['nodes']:
        points.extend(get_leafs(child))
    return points

def cut_tree(tree, threshold):
    if tree["type"] == "leaf":
        return [[tree["data"]]]
    if tree['height'] <= threshold:
        return [get_leafs(tree)]

    clusters = []
    for child in tree["nodes"]:
        clusters.extend(cut_tree(child, threshold))
    return clusters
def labels_from_cut_clusters(data, cut_clusters):
    labels = [-1] * len(data)

    for cluster_id, cluster in enumerate(cut_clusters):
        for point in cluster:
            for i, original_point in enumerate(data):
                if labels[i] == -1 and np.allclose(original_point, point):
                    labels[i] = cluster_id
                    break

    return labels