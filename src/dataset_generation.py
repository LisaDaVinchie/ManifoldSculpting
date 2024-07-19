import numpy as np
import os

class swissRoll:
    def __init__(self, N: int):
        """Initializes the swiss roll dataset generator.

        Args:
            N (int): number of points in the dataset.
        """
        self.N: int = N
        self.t = 8 * np.arange(N) / N + 2
        self.y = np.random.uniform(-6, 6, self.N)

    def swissRoll3D(self):
        """Generates a 3D swiss roll dataset with N samples.

        Returns:
            MatrixLike: 3D swiss roll dataset.
        """
        dataset = np.ndarray((self.N, 3))

        
        dataset[:, 1] = self.y

        dataset[:, 0] = self.t * np.sin(self.t)
        dataset[:, 2] = self.t * np.cos(self.t)

        return dataset
    
    def swissRoll2D(self):
        """Generates the 2D version of the swiss roll dataset with N samples.

        Returns:
            MatrixLike: 2D swiss roll dataset.
        """
        dataset = np.ndarray((self.N, 2))

        dataset[:, 0] = 0.5 * (np.arcsinh(self.t) + self.t  * np.sqrt(self.t ** 2 + 1))
        dataset[:, 1] = self.y

        return dataset

N_points = [250, 300, 400, 500, 600, 700, 800, 1000, 2000]

data_folder = '../data/SwissRoll3D/'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

for N in N_points:
    swissroll = swissRoll(N)
    swissroll3D = swissroll.swissRoll3D()
    swissroll2D = swissroll.swissRoll2D()

    np.save(f'N_{N}.npy', swissroll3D)
    np.save(f'N_{N}.npy', swissroll2D)