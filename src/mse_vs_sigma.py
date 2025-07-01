import numpy as np
import ManifoldSculpting as ms
import time
from pathlib import Path
from utils import parse_paths

def main():

    paths = parse_paths()

    data_folder = Path(paths["dataset3d"])
    destination_folder = Path(paths["results_scalefactors"])
    destination_folder.mkdir(parents=True, exist_ok=True)

    dataset_path = data_folder / "N_1000.npy"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist. Please ensure the dataset is generated.")

    sigma = [0.80, 0.85, 0.99]
    sigma = np.array(sigma, dtype=np.float64)

    n_comps = 2
    n_neighbors = 10

    elapsed_time = []   

    X = np.load(dataset_path)

    for i in range(len(sigma)):
        start_time = time.time()
        model = ms.ManifoldSculpting(n_neighbors = n_neighbors, n_components = n_comps, iterations = 100, max_iter_no_change = 50, sigma = sigma[i])
        X_MS = model.fit(X)
        elapsed_time.append(time.time() - start_time)
        np.savetxt(destination_folder / f"MS_{i}.npy", X_MS)

    np.savetxt(destination_folder / "MS_time.npy", elapsed_time)
    
if __name__ == "__main__":
    main()
