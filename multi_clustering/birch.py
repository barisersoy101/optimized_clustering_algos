import pandas as pd
import numpy as np
from sklearn.cluster import Birch
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

def compute_average_distance(df, column_name):
    embeddings = np.vstack(df[column_name].to_numpy())
    return np.mean(pdist(embeddings, metric='euclidean'))

def birch_clustering(df, column_name, n_clusters=None, num_trials=100):
    avg_distance = compute_average_distance(df, column_name)
    threshold_values = np.linspace(avg_distance * 0.5, avg_distance * 1.5, num_trials)
    best_score = -1
    best_labels = None

    for threshold in threshold_values:
        birch = Birch(threshold=threshold, n_clusters=n_clusters)
        labels = birch.fit_predict(np.vstack(df[column_name].to_numpy()))
        if len(set(labels)) > 1:
            score = silhouette_score(np.vstack(df[column_name].to_numpy()), labels)
            if score > best_score:
                best_score = score
                best_labels = labels

    df['Cluster'] = best_labels
    return df