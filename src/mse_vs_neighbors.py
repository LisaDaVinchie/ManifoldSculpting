import numpy as np
import ManifoldSculpting as ms
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import time
import os

# Load the dataset

X = np.load('../data/SwissRoll3D/N_2000.npy')

data_folder = '../data/results_SwissRoll/'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Define the number of neighbors to test
neighbors = [20, 28, 40, 57, 80]

n_comps = 2

MS_time = []
Isomap_time = []
LLE_time = []

for n in neighbors:

    start_time = time.time()
    isomap = Isomap(n_neighbors=n, n_components=n_comps, metric='euclidean')
    X_isomap = isomap.fit_transform(X)
    Isomap_time.append(time.time() - start_time)
    np.savetxt(data_folder + "Isomap_" + str(n) + ".npy", X_isomap)

    start_time = time.time()
    LLE = LocallyLinearEmbedding(n_neighbors=n, n_components=n_comps)
    X_LLE = LLE.fit_transform(X)
    LLE_time.append(time.time() - start_time)
    np.savetxt(data_folder + "LLE_" + str(n) + ".npy", X_LLE)

    start_time = time.time()
    MS = ms.ManifoldSculpting(n_neighbors=n, n_components=n_comps, iterations=500, max_iter_no_change=50, sigma=0.98)
    X_MS = MS.fit(X)
    MS_time.append(time.time() - start_time)
    np.savetxt(data_folder + "MS_" + str(n) + ".npy", X_MS)
    
    print(f"Done with {n} neighbors")

np.savetxt(data_folder + "MS_time.npy", MS_time)
np.savetxt(data_folder + "Isomap_time.npy", Isomap_time)
np.savetxt(data_folder + "LLE_time.npy", LLE_time)
    