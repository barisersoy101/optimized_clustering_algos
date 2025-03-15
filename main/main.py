import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import yaml
import pandas as pd
from multi_clustering import (
    preprocess_text_column,
    perform_agglomerative_clustering_with_cosine,
    optimal_kmeans_clustering,
    gmm_clustering,
    mean_shift_clustering,
    affinity_propagation_clustering,
    birch_clustering,
    spectral_clustering,
    optimize_optics,
    merge_clustering_results
)

# ---------------------- CONFIG & UTILITIES ---------------------- #

def load_and_preprocess_data(config, text_column):
    """Load and preprocess the data."""
    df = pd.read_csv(config['input_csv'])
    return preprocess_text_column(df, text_column, config)

def load_config(path='config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def run_clustering(df, algorithm, params):
    """Execute clustering algorithm based on the config."""
    clustering_functions = {
        'kmeans': optimal_kmeans_clustering,
        'gmm': gmm_clustering,
        'agglomerative': perform_agglomerative_clustering_with_cosine,
        'affinity': affinity_propagation_clustering,
        'birch': birch_clustering,
        'spectral': spectral_clustering,
        'optics': optimize_optics
    }
    
    if algorithm not in clustering_functions:
        raise ValueError(f"Clustering algorithm '{algorithm}' is not supported.")
    
    return clustering_functions[algorithm](df, 'sbert_embeddings', **params)

def merge_results(clustered_results, id_col, min_common):
    """Merge clustering results."""
    return merge_clustering_results(
        id_col=id_col,
        min_common=min_common,
        **clustered_results
    )

def save_results(df, output_csv):
    """Save the final DataFrame to a CSV file."""
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# ---------------------- MAIN FUNCTION ---------------------- #

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamic Clustering and Merging Tool")
    parser.add_argument('--min_common', type=int, default=3, help='Minimum common clusters for merging.')
    parser.add_argument('--id_col', type=str, required=True, help='Unique identifier column name.')
    parser.add_argument('--text_column', type=str, required=True, help='Text column to be used.')
    parser.add_argument('--algorithms', nargs='+', required=True, help='List of clustering algorithms to use.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file.')  # Added input_file argument
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)

    # Preprocess the data
    df = load_and_preprocess_data(config, args.text_column)

    # Run selected clustering algorithms
    clustered_results = {
        algo: run_clustering(df, algo, config['clustering_algorithms'][algo])
        for algo in args.algorithms
    }

    # Merge and save the results
    final_df = merge_results(clustered_results, args.id_col, args.min_common)
    save_results(final_df, config['output_csv'])

if __name__ == '__main__':
    main()
