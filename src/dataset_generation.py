import numpy as np
from pathlib import Path
import json
import argparse

def main():
    N_points = [250, 300, 400, 500, 600, 700, 800, 1000, 2000]
    
    parser = argparse.ArgumentParser(description='Generate Swiss Roll dataset.')
    parser.add_argument('--paths', type=Path, required=True,  help='Json file containing paths for dataset storage.')
    args = parser.parse_args()
    
    paths_path = Path(args.paths)
    if not paths_path.exists():
        raise FileNotFoundError(f"Paths file {paths_path} does not exist. Please provide a valid path.")
    
    with open(paths_path, 'r') as file:
        paths = json.load(file)

    data_folder_2d = Path(paths["dataset2d"])
    data_folder_3d = Path(paths["dataset3d"])
    data_folder_2d.mkdir(parents=True, exist_ok=True)
    data_folder_3d.mkdir(parents=True, exist_ok=True)

    for N in N_points:
        swissroll = swissRoll(N)
        swissroll3D = swissroll.generate3D()
        swissroll2D = swissroll.generate2D()

        np.save(data_folder_3d / f'N_{N}.npy', swissroll3D)
        np.save(data_folder_2d / f'N_{N}.npy', swissroll2D)
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
    
if __name__ == '__main__':
    main()
    print('Swiss roll dataset generated successfully.')