import numpy as np
import scipy.integrate as integrate
from scipy.special import ellipkinc

class swissRoll:
    def __init__(self, N: int):
        self.N: int = N
        self.t: np.ndarray = 8 * np.arange(N) / N + 2
        self.y = np.random.uniform(-6, 6, self.N)

    def swissRoll3D(self):
        """Generates a swiss roll dataset with N samples."""
        dataset = np.ndarray((self.N, 3))

        
        dataset[:, 1] = self.y
        # N_inv = 1 / self.N

        # t = 8 * np.arange(self.N) * N_inv + 2

        dataset[:, 0] = self.t * np.sin(self.t)
        dataset[:, 2] = self.t * np.cos(self.t)

        return dataset
    
    def swissRoll2D(self):
        dataset = np.ndarray((self.N, 2))

        dataset[:, 0] = 0.5 * (np.arcsinh(self.t) + self.t  * np.sqrt(self.t ** 2 + 1))
        dataset[:, 1] = self.y

        return dataset

class sCurve:
    def __init__(self, N: int):
        self.N: int = N
        self.t = (2.2 * np.arange(N) - 0.1) * np.pi / N

    def sCurve3D(self):
        """Generates a S-curve dataset with N samples."""
        dataset = np.ndarray((self.N, 3))


        dataset[:, 0] = self.t
        dataset[:, 1] = np.sin(self.t)
        dataset[:, 2] = np.random.uniform(0, 2, self.N)

        return dataset


    def sCurve2D(self):
        dataset = np.ndarray((self.N, 2))
        dataset[:, 0] = np.sqrt(2) * ellipkinc(self.t, 0.5)
        dataset[:, 1] = np.sin(self.t)

        return dataset