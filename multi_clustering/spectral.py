import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

def spectral_clustering(df, column_name, min_clusters=2, max_clusters=10, step=1, affinity='nearest_neighbors', assign_labels='discretize'):
    embeddings = np.vstack(df[column_name].to_numpy())
    best_score = -1
    best_labels = None

    for k in range(min_clusters, max_clusters + 1, step):
        clustering = SpectralClustering(n_clusters=k, affinity=affinity, assign_labels=assign_labels, random_state=42)
        labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    df['Cluster'] = best_labels
    return df