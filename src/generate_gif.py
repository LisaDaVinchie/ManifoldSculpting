from pathlib import Path
import imageio.v3 as iio

def generate_gif(gif_path: Path, images_path: list[Path], duration: float = 0.5):
    """Generate a gif showing the space evolution during manifold sculpting

    Args:
        gif_path (Path): path where the gif will be stored
        images_path (list[Path]): folder containing the images to include in the gif
        duration (float, optional): gif duration. Defaults to 0.5.
    """
    print("Reading images for GIF...")
    images = []
    for path in images_path:
        images.append(iio.imread(path))
    print(f"Images read: {len(images)}\n")

    # Save as GIF
    print(f"Saving GIF to {gif_path}...")
    iio.imwrite(gif_path, images, duration=duration, loop=0)
    print(f"GIF saved to {gif_path}")