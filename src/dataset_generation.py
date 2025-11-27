import numpy as np
from pathlib import Path

class swissRoll:
    def __init__(self, N: int):
        """Initializes the swiss roll dataset generator.

        Args:
            N (int): number of points in the dataset.
        """
        self.N: int = N
        self.t = 8 * np.arange(N) / N + 2
        self.y = np.random.uniform(-6, 6, self.N)

    def generate3D(self) -> np.ndarray:
        """Generates a 3D swiss roll dataset with N samples.

        Returns:
            MatrixLike: 3D swiss roll dataset.
        """
        dataset = np.ndarray((self.N, 3))

        
        dataset[:, 1] = self.y

        dataset[:, 0] = self.t * np.sin(self.t)
        dataset[:, 2] = self.t * np.cos(self.t)

        return dataset
    
    def generate2D(self) -> np.ndarray:
        """Generates the 2D version of the swiss roll dataset with N samples.

        Returns:
            MatrixLike: 2D swiss roll dataset.
        """
        dataset = np.ndarray((self.N, 2))

        dataset[:, 0] = 0.5 * (np.arcsinh(self.t) + self.t * np.sqrt(self.t ** 2 + 1))
        dataset[:, 1] = self.y

        return dataset
    
    def save(self, dataset: np.ndarray, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f'N_{self.N}.npy', dataset)
    
    def generate_and_save(self, folder2d: Path, folder3d: Path):
        """Saves the generated datasets to the specified folder.

        Args:
            folder (Path): folder to save the datasets.
        """
        swissroll3D = self.generate3D()
        swissroll2D = self.generate2D()

        np.save(folder3d / f'N_{self.N}.npy', swissroll3D)
        np.save(folder2d / f'N_{self.N}.npy', swissroll2D)