import ManifoldSculpting as ms
from dataset_generation import swissRoll
from pathlib import Path

def main():
    data_folder_3d = Path("./data/datasets/3d/")
    data_folder_2d = Path("./data/datasets/2d/")
    checkpoint_folder = Path("./data/checkpoints/")
    figs_subfolder = "figs/"
    
    N = 800  # Number of points in the dataset
    n_neighbors = 10  # Number of neighbors for the manifold sculpting algorithm
    n_components = 2  # Number of components for the manifold sculpting algorithm
    max_iter_no_change = 50  # Maximum iterations without change
    n_iterations = 100  # Total number of iterations for the algorithm
    save_every = 10  # Save every n iterations

    print("Generating swiss roll dataset...")
    swissroll = swissRoll(N)
    X_3d = swissroll.generate3D()
    X_2d = swissroll.generate2D()
    swissroll.save(X_3d, data_folder_3d)
    swissroll.save(X_2d, data_folder_2d)
    print(f"Swiss roll dataset with {N} points generated and saved.\n")
    
    destination_folder = find_next_available_index(checkpoint_folder)
    figs_folder = destination_folder / figs_subfolder
    figs_folder.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint folder created at: {destination_folder}, figures saved to {figs_folder}\n")

    print("Starting manifold sculpting...")
    model = ms.ManifoldSculpting(n_neighbors=n_neighbors,
                                n_components=n_components,
                                iterations=n_iterations,
                                max_iter_no_change=max_iter_no_change)

    X_MS = model.fit(X_3d, folder = destination_folder, checkpoint_interval = save_every)
    print(f"Manifold sculpting completed. Transformed data shape: {X_MS.shape}")

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
    main()
