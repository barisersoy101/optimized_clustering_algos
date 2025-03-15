import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def gmm_clustering(df, column_name, min_clusters=2, max_clusters=10, step_size=1, random_state=42):
    embeddings = np.vstack(df[column_name].to_numpy())

    best_score = -1
    best_labels = None

    for k in range(min_clusters, max_clusters + 1, step_size):
        gmm_model = GaussianMixture(n_components=k, random_state=random_state)
        cluster_labels = gmm_model.fit_predict(embeddings)
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(embeddings, cluster_labels)
            if score > best_score:
                best_score = score
                best_labels = cluster_labels

    df['Cluster'] = best_labels
    return df