import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def perform_agglomerative_clustering_with_cosine(df, column_name, similarity_threshold=0.8, min_cluster_size=4):
    def compute_cluster_centroids(embedding_matrix, labels):
        unique_labels = set(labels)
        unique_labels.discard(-1)
        centroids = {label: np.mean(embedding_matrix[labels == label], axis=0) for label in unique_labels}
        return centroids

    def merge_clusters(labels, centroids, similarity_threshold):
        cluster_ids = list(centroids.keys())
        cluster_vectors = np.array([centroids[label] for label in cluster_ids])
        similarity_matrix = cosine_similarity(cluster_vectors)

        merged_labels = labels.copy()
        for i, label_1 in enumerate(cluster_ids):
            for j, label_2 in enumerate(cluster_ids):
                if i < j and similarity_matrix[i, j] >= similarity_threshold:
                    merged_labels[merged_labels == label_2] = label_1
        return merged_labels

    embedding_matrix = np.vstack(df[column_name].values)
    distance_matrix = squareform(pdist(embedding_matrix, metric='cosine'))
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=1 - similarity_threshold)
    labels = clustering.fit_predict(distance_matrix)

    centroids = compute_cluster_centroids(embedding_matrix, labels)
    merged_labels = merge_clusters(labels, centroids, similarity_threshold)

    cluster_counts = pd.Series(merged_labels).value_counts()
    noise_clusters = cluster_counts[cluster_counts < min_cluster_size].index
    merged_labels = np.array([-1 if label in noise_clusters else label for label in merged_labels])

    df['Cluster'] = merged_labels
    return df
