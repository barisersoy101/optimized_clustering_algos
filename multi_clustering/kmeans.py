import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def optimal_kmeans_clustering(df, column_name, min_clusters=2, max_clusters=10, step_size=1, random_state=42):
    embeddings = np.vstack(df[column_name].to_numpy())
    
    best_score = -1
    best_labels = None
    best_k = None

    for k in range(min_clusters, max_clusters + 1, step_size):
        kmeans_model = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = kmeans_model.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        if score > best_score:
            best_score = score
            best_labels = cluster_labels
            best_k = k

    df['Cluster'] = best_labels
    return df