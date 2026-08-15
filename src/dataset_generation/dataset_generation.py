"""Generate an n_points points dataset suitable for the algorithm test, i.e. a dataset
of which we know both the 3D and desired 2D form.

As for the paper, I will provide algorithms to generate:

- Swiss Roll

"""

from pathlib import Path
from argparse import ArgumentParser

import numpy as np

class SwissRoll:
    """Generate an n_points points Swiss Roll dataset"""
    def __init__(self, n_points: int):
        """Initializes the swiss roll dataset generator.

        Args:
            N (int): number of points in the dataset.
        """
        self.n_points: int = n_points
        self.t = 8 * np.arange(n_points) / n_points + 2
        self.y = np.random.uniform(-6, 6, self.n_points)

    def generate_3d(self) -> np.ndarray:
        """Generates a 3D swiss roll dataset with N samples.

        Returns:
            np.ndarray: 3D swiss roll dataset, as a N x 3 matrix.
        """
        dataset = np.ndarray((self.n_points, 3))


        dataset[:, 1] = self.y

        dataset[:, 0] = self.t * np.sin(self.t)
        dataset[:, 2] = self.t * np.cos(self.t)

        return dataset

    def generate_2d(self) -> np.ndarray:
        """Generates the 2D version of the swiss roll dataset with N samples.

        Returns:
            np.ndarray: 2D swiss roll dataset, as a N x 2 matrix.
        """
        dataset = np.ndarray((self.n_points, 2))

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

    def generate_and_save(
        self,
        folder: Path, overwrite: bool = False
    ):
        """Generates and saves the 3D and 2D swiss roll datasets, if needed, and returns them.

        Args:
            folder (Path): folder to save the datasets.
            overwrite (bool, optional): whether to overwrite existing files. Defaults to False.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: The generated 3D and 2D swiss roll datasets.
        """

        folder.mkdir(parents=True, exist_ok=True)
        filename_3d = folder / '3d.npy'
        filename_2d = folder / '2d.npy'

        if not overwrite and filename_3d.exists() and filename_2d.exists():
            print(
                f"Files {filename_3d} and {filename_2d} already exist. "
                + "Importing existing datasets."
            )

        swissroll_3d = self.generate_3d()
        swissroll_2d = self.generate_2d()

        self.save(filename_3d, swissroll_3d)
        self.save(filename_2d, swissroll_2d)

        print(f"Datasets saved to {filename_3d} and {filename_2d}")

if __name__ == '__main__':

    p = ArgumentParser()

    p.add_argument("--points", required=True, type=int, help="Number of points in the dataset")
    p.add_argument(
        "--savedir", required=True, type=Path,
        help="Name of the folder containing the 2d and 3d datasets."
    )

    args = p.parse_args()

    print("Generating swiss roll dataset...")
    swissroll = SwissRoll(args.points)
    swissroll.generate_and_save(folder=args.savedir, overwrite=False)
