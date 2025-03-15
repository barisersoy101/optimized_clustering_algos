import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score

def optimize_optics(df, column_name, min_cluster_size=5):
    xi_values = np.linspace(0.001, 0.05, 50)
    embeddings = np.vstack(df[column_name].to_numpy())

    best_score = -1
    best_labels = None

    for xi in xi_values:
        optics = OPTICS(min_samples=5, xi=xi, min_cluster_size=min_cluster_size)
        labels = optics.fit_predict(embeddings)
        if len(set(labels)) > 1:
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

    df['Cluster'] = best_labels
    return df
