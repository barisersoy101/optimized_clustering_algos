import re
from collections import Counter
from nltk.corpus import stopwords

nltk_stopwords = set(stopwords.words('english'))

def merge_clustering_results(id_col, min_common=None, **cluster_dfs):
    """
    Merge multiple clustering results dynamically.

    Parameters:
        id_col (str): The identifier column.
        min_common (int): Minimum number of common clusters for merging.
        **cluster_dfs: Clustering DataFrames with their names.

    Returns:
        pd.DataFrame: Merged DataFrame with assigned clusters and summaries.
    """
    if not cluster_dfs:
        raise ValueError("At least one clustering DataFrame must be provided.")

    # Initialize with the first DataFrame and keep cleaned_text
    first_df = next(iter(cluster_dfs.values()))
    df = first_df[[id_col, 'cleaned_text']].drop_duplicates(subset=[id_col]).copy()

    clustering_cols = []

    # Merge clustering results dynamically
    for cluster_name, clustering_df in cluster_dfs.items():
        cluster_col_name = f'{cluster_name}_cluster'
        clustering_df = clustering_df[[id_col, clustering_df.columns[-1]]].drop_duplicates(subset=[id_col])

        # Perform merge and rename the cluster column
        df = df.merge(clustering_df, on=id_col, how='left')
        df = df.rename(columns={clustering_df.columns[-1]: cluster_col_name})
        clustering_cols.append(cluster_col_name)

    # Create mapping of identifier to cluster assignments
    app_clusters = df.set_index(id_col)[clustering_cols].to_dict(orient='index')

    assigned_clusters = {}
    current_cluster_id = 1
    processed = set()

    # Iterate and assign merged clusters
    for item_id, clusters in app_clusters.items():
        if item_id in processed:
            continue

        group = {item_id}

        for other_id, other_clusters in app_clusters.items():
            if other_id == item_id or other_id in processed:
                continue

            # Count how many clusters match
            common = sum(clusters[algo] == other_clusters[algo] for algo in clustering_cols)
            
            if common >= min_common:
                group.add(other_id)

        # Assign the same cluster ID to the group
        for member in group:
            assigned_clusters[member] = current_cluster_id
            processed.add(member)
        
        current_cluster_id += 1

    # Assign merged cluster IDs back to the DataFrame
    df['merged_cluster'] = df[id_col].map(assigned_clusters)

    # Generate cluster summaries using cleaned_text
    summaries = {}
    for cluster_id, group_df in df.groupby('merged_cluster'):
        descriptions = ' '.join(group_df['cleaned_text'].dropna())
        words = re.findall(r'\b\w+\b', descriptions.lower())
        filtered_words = [word for word in words if word not in nltk_stopwords]
        common_words = [word for word, _ in Counter(filtered_words).most_common(10)]
        summaries[cluster_id] = ', '.join(common_words)

    # Map summaries back to the DataFrame
    df['cluster_summary'] = df['merged_cluster'].map(summaries)

    # Only keep relevant columns
    final_cols = [id_col, 'cleaned_text'] + clustering_cols + ['merged_cluster', 'cluster_summary']
    return df[final_cols]