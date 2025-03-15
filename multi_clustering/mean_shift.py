import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from scipy.spatial.distance import pdist

def compute_bandwidth(df, column_name):
    embeddings = np.vstack(df[column_name].to_numpy())
    pairwise_distances = pdist(embeddings, metric='euclidean')
    return np.mean(pairwise_distances)

def mean_shift_clustering(df, column_name):
    embeddings = np.vstack(df[column_name].to_numpy())
    bandwidth = compute_bandwidth(df, column_name)
    mean_shift = MeanShift(bandwidth=bandwidth)
    cluster_labels = mean_shift.fit_predict(embeddings)
    df['Cluster'] = cluster_labels
    return df