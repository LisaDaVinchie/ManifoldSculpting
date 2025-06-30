import ManifoldSculpting as ms
import numpy as np
from pathlib import Path
from utils import parse_paths

def main():
    paths = parse_paths()
        
    data_folder_3d = Path(paths["dataset3d"])
    N = 2000  # Number of points in the dataset
    n_neighbors = 20  # Number of neighbors for the manifold sculpting algorithm
    n_components = 2  # Number of components for the manifold sculpting algorithm
    max_iter_no_change = 50  # Maximum iterations without change
    n_iterations = 500  # Total number of iterations for the algorithm
    save_every = 10  # Save every n iterations

    dataset_path = data_folder_3d / f'N_{N}.npy'
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist. Please provide a valid dataset.")

    destination_folder = Path(paths["checkpoints_next"])

    if not destination_folder.parent.exists():
        raise FileNotFoundError(f"Destination folder {destination_folder.parent} does not exist. Please create it before running the script.")

    destination_folder.mkdir(parents=True, exist_ok=False)


    X = np.load(dataset_path)
    print(f"Loaded dataset from {dataset_path}")

    model = ms.ManifoldSculpting(n_neighbors=n_neighbors,
                                n_components=n_components,
                                iterations=n_iterations,
                                max_iter_no_change=max_iter_no_change,)

    X_MS = model.fit(X, save_checkpoints = True, folder = destination_folder, checkpoint_interval = save_every)

    # last_epoch = model.elapsed_epochs

    # np.save(destination_folder + f"checkpoint_{last_epoch}.npy", X_MS)

if __name__ == "__main__":
    main()
