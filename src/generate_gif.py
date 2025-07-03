import numpy as np
from pathlib import Path
from utils import parse_paths
import matplotlib.pyplot as plt
import imageio.v3 as iio

def main():
    paths = parse_paths()

    destination_fig_folder = Path(paths["checkpoints_fig_next"])
    if not destination_fig_folder.parent.exists():
        raise FileNotFoundError(f"Destination folder {destination_fig_folder.parent} does not exist. Please create it before running the script.")
    destination_fig_folder.mkdir(parents=True, exist_ok=True)

    files_folder = Path(paths["checkpoints"])
    if not files_folder.parent.exists():
        raise FileNotFoundError(f"Destination folder {files_folder} does not exist. Please provide a valid path.")

    files_check = sorted(files_folder.glob('checkpoint_*.npy'))
    if not files_check:
        raise FileNotFoundError(f"No checkpoint files found in {files_folder}. Please ensure the path is correct and files exist.")

    gif_path = Path(paths["gif_path"])
    if not gif_path.parent.exists():
        raise FileNotFoundError(f"Destination folder {gif_path.parent} does not exist. Please create it before running the script.")

    images_path = [destination_fig_folder / f"{file.stem}.png" for file in files_check]
    if all(image_path.exists() for image_path in images_path):
        print("All images already exist. Skipping plot generation.")
        
    else:
        # Generate plots for each checkpoint
        print("Generating plots for each checkpoint...")
        for i, file in enumerate(files_check):
            X = np.load(file)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 1])

            plt.savefig(images_path[i])

            plt.close()

        print(f"Plots saved to {destination_fig_folder}\n")

    print("Reading images for GIF...")
    images = []
    for path in images_path:
        images.append(iio.imread(path))
    print(f"Images read: {len(images)}\n")

    # Save as GIF
    print(f"Saving GIF to {gif_path}...")
    iio.imwrite(gif_path, images, duration=0.5, loop=0)
    print(f"GIF saved to {gif_path}")