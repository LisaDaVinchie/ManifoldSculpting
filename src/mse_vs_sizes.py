import numpy as np
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import ManifoldSculpting as ms
import os
import time

data_folder = "../data/sCurve3D/"

destination_folder = "../data/results_sCurve/k14/"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

sizes = [250, 500, 1000, 2000, 4000]

n_comps = 2
n_neighbors = 14

elapsed_time = []

for s in sizes:
    X = np.load(data_folder + f"N_{s}.npy")

    X_isomap = Isomap(n_neighbors=n_neighbors, n_components=n_comps).fit_transform(X)
    np.savetxt(destination_folder + f"isomap_{s}.npy", X_isomap)
               

    X_LLE = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_comps).fit_transform(X)
    np.savetxt(destination_folder + f"LLE_{s}.npy", X_LLE)

    start_time = time.time()
    model = ms.ManifoldSculpting(n_neighbors = n_neighbors, n_components = n_comps, iterations = 500, max_iter_no_change = 50, sigma = 0.98)
    X_MS = model.fit(X)
    elapsed_time.append(time.time() - start_time)
    np.savetxt(destination_folder + f"MS_{s}.npy", X_MS)
    
    print(f"size = {s} done")

np.savetxt(destination_folder + "MS_time.npy", elapsed_time)
