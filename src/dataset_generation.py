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
            np.ndarray: 3D swiss roll dataset.
        """
        dataset = np.ndarray((self.N, 3))

        
        dataset[:, 1] = self.y

        dataset[:, 0] = self.t * np.sin(self.t)
        dataset[:, 2] = self.t * np.cos(self.t)

        return dataset
    
    def generate2D(self) -> np.ndarray:
        """Generates the 2D version of the swiss roll dataset with N samples.

        Returns:
            np.ndarray: 2D swiss roll dataset.
        """
        dataset = np.ndarray((self.N, 2))

        dataset[:, 0] = 0.5 * (np.arcsinh(self.t) + self.t * np.sqrt(self.t ** 2 + 1))
        dataset[:, 1] = self.y

        return dataset
    
    def save(self, filename: Path, dataset: np.ndarray) -> None:
        """Saves the dataset to the specified path

        Args:
            filename (Path): path to the file to save
            dataset (np.ndarray): file to save
        """
        np.save(filename, dataset)
    
    def generate_and_save(self, dataset_folder: Path, overwrite: bool = False, subfolder_2d: str = "2d", subfolder_3d: str = "3d", file_name_prefix: str = "N"):
        """Generates and saves the 3D and 2D swiss roll datasets, if needed, and returns them.

        Args:
            folder (Path): folder to save the datasets.
            overwrite (bool, optional): whether to overwrite existing files. Defaults to False.
            subfolder_2d (str, optional): subfolder name for 2D dataset. Defaults to "2d".
            subfolder_3d (str, optional): subfolder name for 3D dataset. Defaults to "3d".
            file_name_prefix (str, optional): prefix for the saved file names. Defaults to "N".
        
        Returns:
            tuple[np.ndarray, np.ndarray]: The generated 3D and 2D swiss roll datasets.
        """

        folder3d = dataset_folder / subfolder_3d
        folder2d = dataset_folder / subfolder_2d
        folder3d.mkdir(parents=True, exist_ok=True)
        folder2d.mkdir(parents=True, exist_ok=True)
        filename_3d = folder3d / f'{file_name_prefix}_{self.N}.npy'
        filename_2d = folder2d / f'{file_name_prefix}_{self.N}.npy'
        
        if not overwrite and filename_3d.exists() and filename_2d.exists():
            print(f"Files {filename_3d} and {filename_2d} already exist. Importing existing datasets.")
            
            swissroll3D = np.load(filename_3d)
            swissroll2D = np.load(filename_2d)
            return swissroll3D, swissroll2D
        
        swissroll3D = self.generate3D()
        swissroll2D = self.generate2D()
        
        self.save(filename_3d, swissroll3D)
        self.save(filename_2d, swissroll2D)
        
        return swissroll3D, swissroll2D