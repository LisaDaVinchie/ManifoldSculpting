import numpy as np
import ManifoldSculpting as ms
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import time
from pathlib import Path
from utils import parse_paths

def main():
    paths = parse_paths()
    
    N = 2000
    neighbors = [20, 28, 40, 57, 80]
    n_comps = 2
    iterations = 800
    max_iter_no_change = 50

    data_folder = Path(paths["dataset3d"])
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder {data_folder} does not exist. Please provide a valid dataset path.")
    destination_folder = Path(paths["results_neighbors"])
    destination_folder.mkdir(parents=True, exist_ok=True)

    dataset_path = data_folder / f"N_{N}.npy"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist. Please ensure the dataset is generated.")

    X = np.load(dataset_path)
    print(f"Loaded dataset from {dataset_path}, shape: {X.shape}")

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    X -= X_mean
    X /= X_std
    print("Dataset normalized: mean and std subtracted.")
    
    times_files = {
        "Isomap": [],
        "LLE": [],
        "MS": []
    }
        

    for n in neighbors:
        print(f"Processing with {n} neighbors...")

        start_time = time.time()
        ISOMAP = Isomap(n_neighbors=n, n_components=n_comps, metric='euclidean')
        X_ISOMAP = ISOMAP.fit_transform(X)
        times_files["Isomap"].append(time.time() - start_time)
        np.savetxt(destination_folder / f"Isomap_{str(n)}.npy", X_ISOMAP)

        start_time = time.time()
        LLE = LocallyLinearEmbedding(n_neighbors=n, n_components=n_comps)
        X_LLE = LLE.fit_transform(X)
        times_files['LLE'].append(time.time() - start_time)
        np.savetxt(destination_folder / f"LLE_{str(n)}.npy", X_LLE)

        start_time = time.time()
        MS = ms.ManifoldSculpting(n_neighbors=n,
                                  n_components=n_comps,
                                  iterations=iterations,
                                  max_iter_no_change=max_iter_no_change)
        X_MS = MS.fit(X)
        times_files["MS"].append(time.time() - start_time)
        np.savetxt(destination_folder / f"MS_{str(n)}.npy", X_MS)
    
    

    np.savetxt(destination_folder / "Isomap_time.npy", times_files["Isomap"])
    np.savetxt(destination_folder / "LLE_time.npy", times_files["LLE"])
    np.savetxt(destination_folder / "MS_time.npy", times_files["MS"])
    print("All computations completed and results saved.")

if __name__ == "__main__":
    main()