import numpy as np
import ManifoldSculpting_v2 as ms
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import time

# Load the dataset

X = np.load('../data/SwissRoll3D/N_2000.npy')

data_folder = '../data/results_SwissRoll/neighbors/'

# Define the number of neighbors to test
neighbors = [20, 28, 40, 57, 80]

n_comps = 2

for n in neighbors:
    isomap = Isomap(n_neighbors=n, n_components=n_comps, metric='euclidean')
    X_isomap = isomap.fit_transform(X)
    np.savetxt(data_folder + "Isomap_" + str(n) + ".npy", X_isomap)

    LLE = LocallyLinearEmbedding(n_neighbors=n, n_components=n_comps)
    X_LLE = LLE.fit_transform(X)
    np.savetxt(data_folder + "LLE_" + str(n) + ".npy", X_LLE)

    start_time = time.time()
    MS = ms.ManifoldSculpting(n_neighbors=n, n_components=n_comps, iterations=800, max_iter_no_change=50, sigma=0.98)
    X_MS = MS.fit(X)
    elapsed_time = time.time() - start_time
    print(X_MS)
    np.savetxt(data_folder + "MS_" + str(n) + ".npy", X_MS)
    np.savetxt(data_folder + "time_" + str(n) + ".npy", elapsed_time)
    print(f"Done with {n} neighbors")
    