import numpy as np
import ManifoldSculpting as ms
import os
import time

data_folder = "../data/SwissRoll3D/"

destination_folder = "../data/results_SwissRoll/scaleFactors/"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

sigma = [0.80, 0.85, 0.99]
sigma = np.array(sigma, dtype=np.float64)

n_comps = 2
n_neighbors = 10

elapsed_time = []   

X = np.load(data_folder + f"N_1000.npy")

for i in range(len(sigma)):
    start_time = time.time()
    model = ms.ManifoldSculpting(n_neighbors = n_neighbors, n_components = n_comps, iterations = 100, max_iter_no_change = 50, sigma = sigma[i])
    X_MS = model.fit(X)
    elapsed_time.append(time.time() - start_time)
    np.savetxt(destination_folder + f"MS_{i}.npy", X_MS)

np.savetxt(destination_folder + "MS_time.npy", elapsed_time)
