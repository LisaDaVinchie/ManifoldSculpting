import argparse
import json
from pathlib import Path
import numpy as np

def parse_paths() -> dict:
    """Parse the paths from the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, required=True, help='Json file containing paths for dataset storage.')
    args = parser.parse_args()

    paths_path = Path(args.paths)
    if not paths_path.exists():
        raise FileNotFoundError(f"Paths file {paths_path} does not exist. Please provide a valid path.")

    with open(paths_path, 'r') as file:
        paths = json.load(file)
    return paths

def pca(data):
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvectors

def rotate_dataset(X_ms):

    X_ms -= np.mean(X_ms, axis=0)
    _, eigenvecs = pca(X_ms)

    rotation_matrix = np.eye(3)
    rotation_matrix[:, :2] = eigenvecs[:, :2]

    # Rotate the data
    return X_ms @ rotation_matrix