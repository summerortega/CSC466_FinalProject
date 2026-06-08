import sys
import numpy as np
import pandas as pd
import math
from evaluation import read_csv, print_cluster_report

#select centroids at random
def select_centroids(df: pd.DataFrame, k:int) -> np.ndarray:
    return df.sample(k, axis=0).to_numpy()


#calculate sum-squared-error
def calc_sse(centroids:np.ndarray, clusters:np.ndarray) -> float:
    new_sse = 0
    for i in range(centroids.shape[0]):
        #sum of all Euclidean distances
        new_sse += np.sum([euc_dist(centroids[i], clusters[j]) ** 2 for j in range(clusters.shape[0])])
    return new_sse


#calculate Euclidean distance between two points
def euc_dist(x:pd.Series, centroid:pd.Series) -> float:
    return math.sqrt(np.sum(np.square(x - centroid)))


#main k-means algorithm
def main(csv_file:str, k:int, threshold:float = 0.025, ground_truth_col=None):
    #read dataframe
    df, ground_truth = read_csv(csv_file, ground_truth_col)
    #select centroids
    centroids = select_centroids(df, k)
    prev_sse = -1
    current_sse = 0
    #while sse difference is significant
    while abs(prev_sse - current_sse) > threshold:
        #set up cluster sum and num_pts arrays. used for finding means to recalculate centroids
        cluster_sums = np.zeros(shape=(k, df.columns.size))
        num_pts = np.zeros(shape=k)
        #set clusters and labels. Labels used for final output, clusters for SSE.
        clusters = np.full(shape=(k, df.shape[0], df.shape[1]), fill_value=0)
        labels = np.full(shape=(df.shape[0]), fill_value=-1)
        # step 3: compute distances for all points and add to appropriate cluster
        for i in range(df.shape[0]):
            #Euclidean distance to every centroid
            dists = np.array([euc_dist(df.iloc[i], centroids[l]) for l in range(k)])
            #assign point to cluster with minimum distance
            cluster = np.argmin(dists)
            cluster_sums[cluster] = cluster_sums[cluster] + df.iloc[i]
            num_pts[cluster] += 1
            clusters[cluster][i] = df.iloc[i].to_numpy(dtype="float32")
            labels[i] = cluster
        #clean clusters of their zero arrays
        final_clusters = []
        zero_arr = np.zeros(shape=(df.shape[1]))
        shape = clusters[0].shape[0]
        for i in range(clusters.shape[0]):
            mask = [np.array_equal(clusters[i][j], zero_arr) for j in range(shape)]
            new_mask = np.logical_not(mask)
            cluster = clusters[i][new_mask]
            final_clusters.append(cluster)
        #recalculate centroids
        centroids = np.array([cluster_sums[l] / num_pts[l] for l in range(k)])
        #calculate centroids
        prev_sse = current_sse
        current_sse = calc_sse(centroids, clusters)
    print(labels)
    print_cluster_report(df.to_numpy(), labels, ground_truth)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 4:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
    elif len(sys.argv) == 5:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    else:
        print("Usage: python3 kmeans.py <csv_file> <k> [<threshold>] [<ground_truth_col>]")
