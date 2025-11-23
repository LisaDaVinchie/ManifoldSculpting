import ManifoldSculpting as ms
import numpy as np
from pathlib import Path

def main():
    data_folder_3d = Path("./data/datasets/3d/")
    N = 800  # Number of points in the dataset
    n_neighbors = 10  # Number of neighbors for the manifold sculpting algorithm
    n_components = 2  # Number of components for the manifold sculpting algorithm
    max_iter_no_change = 50  # Maximum iterations without change
    n_iterations = 100  # Total number of iterations for the algorithm
    save_every = 10  # Save every n iterations

    dataset_path = data_folder_3d / f'N_{N}.npy'
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist. Please provide a valid dataset.")

    idx = 1
    checkpoint_folder = Path("./data/checkpoints/")
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    
    destination_folder = Path(checkpoint_folder / f"trial_{idx}/")
    while destination_folder.exists():
        idx += 1
        destination_folder = Path(checkpoint_folder / f"trial_{idx}/")
    destination_folder.mkdir(parents=True, exist_ok=False)
    
    X = np.load(dataset_path)
    print(f"Loaded dataset from {dataset_path}")

    model = ms.ManifoldSculpting(n_neighbors=n_neighbors,
                                n_components=n_components,
                                iterations=n_iterations,
                                max_iter_no_change=max_iter_no_change)

    X_MS = model.fit(X, folder = destination_folder, checkpoint_interval = save_every)
    
    print(f"Manifold sculpting completed. Transformed data shape: {X_MS.shape}")

if __name__ == "__main__":
    main()
