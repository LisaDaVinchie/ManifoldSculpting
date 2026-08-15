"""Run the manifold sculpting algorithm"""

from pathlib import Path
from argparse import ArgumentParser

import numpy as np

import ManifoldSculpting as ms
from generate_gif import generate_gif

def main(filepath: Path, savedir: Path):
    n_neighbors = 10  # Number of neighbors for the manifold sculpting algorithm
    n_components = 2  # Number of components for the manifold sculpting algorithm
    max_iter_no_change = 50  # Maximum iterations without change
    n_iterations = 100  # Total number of iterations for the algorithm
    save_every = 10  # Save every n iterations
    generate_gif_flag = True  # Whether to generate a GIF of the evolution

    if not filepath.exists():
        raise ValueError(f"Dataset {filepath} does not exist!")

    figs_subfolder = "figs/"

    print(f"Loading {filepath}...\n")
    dataset_3d = np.load(filepath)
    print(f"Dataset {filepath} loaded.\n")

    destination_folder = find_next_available_index(savedir)
    figs_folder = destination_folder / figs_subfolder
    figs_folder.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint folder created at: {destination_folder}, figures saved to {figs_folder}\n")

    print("Starting manifold sculpting...")
    model = ms.ManifoldSculpting(n_neighbors=n_neighbors,
                                n_components=n_components,
                                iterations=n_iterations,
                                max_iter_no_change=max_iter_no_change)

    X_MS = model.fit(dataset_3d, folder = destination_folder, checkpoint_interval = save_every, figs_subfolder=figs_subfolder, savefig=generate_gif_flag)
    print(f"Manifold sculpting completed. Transformed data shape: {X_MS.shape}\n")

    if generate_gif_flag:
        try:
            print("Generating GIF from saved figures...")
            fig_paths = sorted(figs_folder.glob("*.png"))
            gif_path = destination_folder / "evolution.gif"
            generate_gif(gif_path, fig_paths)
            print("GIF generation completed.")
        except Exception as e:
            print(f"An error occurred while generating the GIF: {e}")

def find_next_available_index(checkpoint_folder: Path, file_name: str = "trial") -> Path:
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    idx = 1
    destination_folder = Path(checkpoint_folder / f"{file_name}_{idx}/")
    while destination_folder.exists():
        idx += 1
        destination_folder = Path(checkpoint_folder / f"{file_name}_{idx}/")
    destination_folder.mkdir(parents=True, exist_ok=False)
    return destination_folder

if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument(
        "--dataset", required=True, type=Path,
        help="Path to the 3d dataset to run Manifold Sculpting on"
    )
    p.add_argument(
        "--savedir", required=True, type=Path,
        help="Path to the folder to save the results to."
    )

    args = p.parse_args()
    main(args.dataset, args.savedir)
