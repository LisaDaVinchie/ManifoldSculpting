import argparse
import json
from pathlib import Path

def parse_paths() -> dict:
    """Parse the paths from the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, required=True, help='Json file containing paths for dataset storage.')
    args = parser.parse_args()

    paths_path = Path(args.paths)
    if not paths_path.exists():
        raise FileNotFoundError(f"Paths file {paths_path} does not exist. Please provide a valid path.")

    with open(paths_path, 'r') as file:
        paths = json.load(file)
    return paths