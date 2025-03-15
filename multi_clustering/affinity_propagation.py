import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation

def affinity_propagation_clustering(df, column_name):
    embeddings = np.vstack(df[column_name].to_numpy())
    affinity = AffinityPropagation()
    cluster_labels = affinity.fit_predict(embeddings)
    df['Cluster'] = cluster_labels
    
    return df