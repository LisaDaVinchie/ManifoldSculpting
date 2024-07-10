import numpy as np
import ManifoldSculpting_v1 as MS
import matplotlib.pyplot as plt

dataset = np.load('../data/SwissRoll3D/N_250.npy')

model = MS.ManifoldSculpting(iterations=10, n_neighbors=5, n_components=2)
X, errors = model.fit_transform(dataset)

np.save('../data/results_SwissRoll/N_250_It_10_Ngbr_5.npy', X)

# plt.plot(errors)
# plt.show()



