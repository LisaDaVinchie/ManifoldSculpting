import numpy as np
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import ManifoldSculpting as ms
import os
import time
from pathlib import Path
from utils import parse_paths

def main():
    paths = parse_paths()

    data_folder = Path(paths["dataset3d"])
    
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder {data_folder} does not exist. Please provide a valid dataset path.")

    destination_folder = Path(paths["results_sizes"])

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    sizes = [250, 500, 1000, 2000, 4000]
    paths = [data_folder / f"N_{s}.npy" for s in sizes]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f"One or more dataset files do not exist in {data_folder}. Please ensure the datasets are generated.")
    

    n_comps = 2
    n_neighbors = 20

    elapsed_time = []

    for s in sizes:
        print(f"Processing dataset with {s} points.")
        X = np.load(data_folder / f"N_{s}.npy")

        X_isomap = Isomap(n_neighbors=n_neighbors, n_components=n_comps).fit_transform(X)
        np.savetxt(destination_folder / f"isomap_{s}.npy", X_isomap)
                

        X_LLE = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_comps).fit_transform(X)
        np.savetxt(destination_folder / f"LLE_{s}.npy", X_LLE)

        start_time = time.time()
        model = ms.ManifoldSculpting(n_neighbors = n_neighbors, n_components = n_comps, iterations = 500, max_iter_no_change = 50, sigma = 0.98)
        X_MS = model.fit(X)
        elapsed_time.append(time.time() - start_time)
        np.savetxt(destination_folder / f"MS_{s}.npy", X_MS)

    np.savetxt(destination_folder + "MS_time.npy", elapsed_time)

if __name__ == "__main__":
    main()
