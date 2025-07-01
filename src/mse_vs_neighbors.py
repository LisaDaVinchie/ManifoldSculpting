import numpy as np
import ManifoldSculpting as ms
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import time
from pathlib import Path
from utils import parse_paths

def main():
    paths = parse_paths()

    data_folder = Path(paths["dataset3d"])
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder {data_folder} does not exist. Please provide a valid dataset path.")
    destination_folder = Path(paths["results_neighbors"])
    destination_folder.mkdir(parents=True, exist_ok=True)

    dataset_path = data_folder / "N_2000.npy"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist. Please ensure the dataset is generated.")

    X = np.load(dataset_path)
    print(f"Loaded dataset from {dataset_path}, shape: {X.shape}")

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    X -= X_mean
    X /= X_std
    print("Dataset normalized: mean and std subtracted.")

    neighbors = [20, 28, 40, 57, 80]

    n_comps = 2

    Isomap_time = []
    LLE_time = []
    MS_time = []

    for n in neighbors:
        print(f"Processing with {n} neighbors...")

        start_time = time.time()
        ISOMAP = Isomap(n_neighbors=n, n_components=n_comps, metric='euclidean')
        X_ISOMAP = ISOMAP.fit_transform(X)
        Isomap_time.append(time.time() - start_time)
        np.savetxt(destination_folder / f"Isomap_{str(n)}.npy", X_ISOMAP)

        start_time = time.time()
        LLE = LocallyLinearEmbedding(n_neighbors=n, n_components=n_comps)
        X_LLE = LLE.fit_transform(X)
        LLE_time.append(time.time() - start_time)
        np.savetxt(destination_folder / f"LLE_{str(n)}.npy", X_LLE)

        start_time = time.time()
        MS = ms.ManifoldSculpting(n_neighbors=n, n_components=n_comps, iterations=800, max_iter_no_change=50)
        X_MS = MS.fit(X)
        MS_time.append(time.time() - start_time)
        np.savetxt(destination_folder / f"MS_{str(n)}.npy", X_MS)
        print(f"Processed {n} neighbors in {MS_time[-1]:.2f} seconds.")

    np.savetxt(destination_folder / "Isomap_time.npy", Isomap_time)
    np.savetxt(destination_folder / "LLE_time.npy", LLE_time)
    np.savetxt(destination_folder / "MS_time.npy", MS_time)
    print("All computations completed and results saved.")

if __name__ == "__main__":
    main()