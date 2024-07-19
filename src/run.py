import ManifoldSculpting as ms
import numpy as np
import os

# Load the dataset
X = np.load('../data/SwissRoll3D/N_2000.npy')

destination_folder = '../data/SwissRoll_checkpoints/k20/'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

model = ms.ManifoldSculpting(n_neighbors=20,
                             n_components=2,
                             iterations=500,
                             max_iter_no_change=50)

X_MS = model.fit(X, save_checkpoints = True, folder = destination_folder, checkpoint_interval = 10)

last_epoch = model.elapsed_epochs

np.save(destination_folder + f"checkpoint_{last_epoch}.npy", X_MS)
