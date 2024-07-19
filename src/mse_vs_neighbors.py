import numpy as np
import ManifoldSculpting as ms
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import time
import os

X = np.load('../data/SwissRoll3D/N_2000.npy')

X_mean = np.mean(X, axis=0)

X = X - X_mean

X_std = np.std(X, axis=0)

X = X / X_std

data_folder = '../data/results_SwissRoll/neighbors/'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

neighbors = [20, 28, 40, 57, 80]

n_comps = 2

Isomap_time = []
LLE_time = []
MS_time = []

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
    MS = ms.ManifoldSculpting(n_neighbors=n, n_components=n_comps, iterations=800, max_iter_no_change=50)
    X_MS = MS.fit(X)
    MS_time.append(time.time() - start_time)
    np.savetxt(data_folder + "MS_" + str(n) + ".npy", X_MS)

np.savetxt(data_folder + "Isomap_time.npy", Isomap_time)
np.savetxt(data_folder + "LLE_time.npy", LLE_time)
np.savetxt(data_folder + "MS_time.npy", MS_time)
    