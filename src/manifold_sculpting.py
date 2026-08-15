"""Manifold sculpting class"""
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils import find_knn, find_mcn, compute_pca, average_neighbor_distance

class ManifoldSculpting:
    """Perform manifold sculpting on the dataset"""

    def __init__(
        self,
        n_neighbors: int = 5, n_components: int = 2,
        iterations: int = 100, sigma: float = 0.99,
        perform_pca: bool = True, max_iter_no_change: int = 30
    ):
        """Used to pass parameters to che class

        Args:
            n_neighbors (int, optional): Number of neighbors for each point. Defaults to 5.
            n_components (int, optional): Number of dimensions to preserve. Defaults to 2.
            iterations (int, optional): Max number of iterations. Defaults to 100.
            sigma (float, optional): Scale factor. Defaults to 0.99.
            perform_pca (bool, optional): Decide if you want to perform PCA. Defaults to True.
            max_iter_no_change (int, optional): Maximum number of iterations with no change in the error. Defaults to 30.
        """
        self.n_neighbors: int = n_neighbors
        self.n_components: int = n_components
        self.n_iterations: int = iterations
        self.sigma: float = sigma
        self.rotate: bool = perform_pca
        self.scale_factor: float = 1

        self.max_iter_no_change: int = max_iter_no_change

    def fit(self, data: np.ndarray, folder: Path = "./", figs_subfolder: str = "figs", checkpoint_interval: int = 10, scale_factor_threshold: float = 0.01, savefig: bool = False) -> np.ndarray:
        """Pass the dataset to transform it into a lower dimension

        Args:
            data (np.ndarray): dataset to transfrom, made as a matrix of shape (n_samples, n_features)
            folder (str, optional): Where to save the checkpoints. Defaults to './'.
            figs_subfolder (str, optional): subfolder to save the figures. Defaults to "figs".
            checkpoint_interval (int, optional): number of epochs between one checkpoint and another one. Defaults to 10.
            scale_factor_threshold (float, optional): scale factor threshold to finish the heat up phase. Defaults to 0.01.
            savefig (bool, optional): whether to save the figures or not. Defaults to False

        Returns:
            MatrixLike: transformed dataset
        """

        self.folder: Path = folder
        n_points: int = data.shape[0]
        self.savefig: bool = savefig
        
        self.figs_folder: Path = folder / figs_subfolder
        
        # 1 - 2) Initialise KNN and MCN and calculate distances and angles
        self.neighbours, self.distances0, self.avg_dist0= find_knn(data, self.n_neighbors)
        self.mcn_index, self.mcn_angles = find_mcn(data, self.neighbours, self.n_neighbors)
        
        self.learning_rate = self.avg_dist0
        
        # 3) Optional: align the data with PCA
        if self.rotate:
            self.pca_data = compute_pca(data)
            self.d_pres = np.arange(self.n_components, dtype=np.int32)
            self.d_scal = np.arange(self.n_components, data.shape[1], dtype=np.int32)
        else:
            cov = np.cov(data.T)
            most_important = np.argsort(-np.diag(cov)).astype(np.int32)
            self.d_pres = most_important[:self.n_components]
            self.d_scal = most_important[self.n_components:]
            self.pca_data = np.copy(data)

        # Save initial state
        self.save_checkpoint(0)
        
        self.epoch: int = 1

        print(f"Starting manifold sculpting with {n_points} points and {self.n_neighbors} neighbors.\n")
        
        print(f"Starting heat up with scale factor {self.scale_factor}.\n")
        mean_error = self._heat_up(checkpoint_interval, scale_factor_threshold, figs_subfolder, savefig)
        print(f"Heat up finished in {self.epoch} epochs. Scale factor is now {self.scale_factor}.\n")

        print(f"Starting manifold sculpting\n")
        epochs_since_improvement = 0
        best_error = np.inf
        
        # 4) Until the stopping criteria is met, perform manifold sculpting steps
        while (self.epoch < self.n_iterations) and (epochs_since_improvement < self.max_iter_no_change):
            print(f"Epoch {self.epoch}...")
            mean_error = self._step()

            # If the error improved, save the best state. Otherwise, increase the counter
            if mean_error < best_error:
                best_error = mean_error
                self.best_data = np.copy(self.pca_data)
                self.best_error = best_error
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                
            if self.epoch % checkpoint_interval == 0:
                print(f"Mean error: {mean_error}, best error: {best_error}, epochs since improvement: {epochs_since_improvement}")
            
            if self.epoch % checkpoint_interval == 0:
                self.save_checkpoint(self.epoch)
                
            self.epoch += 1
            print()

        self.elapsed_epochs = self.epoch
        self.last_error = mean_error

        return self.pca_data

    def _heat_up(self, checkpoint_interval, scale_factor_threshold, figs_subfolder, savefig):
        while self.scale_factor > scale_factor_threshold:
            if self.epoch % checkpoint_interval == 0:
                print(f"Epoch {self.epoch}, scale factor: {self.scale_factor}")
            mean_error = self._step()
            self.epoch += 1

            if self.epoch % checkpoint_interval == 0:
                self.save_checkpoint(self.epoch)
        return mean_error

    def save_checkpoint(self, epoch: int, basename: str = "checkpoint", extension: str = "npy", fig_extension: str = "png"):
        filename = f"{basename}_{epoch:04d}"
        np.save(self.folder / filename, self.pca_data)
        print(f"Checkpoint saved at {self.folder / f'{filename}.{extension}'}")
        
        if self.savefig:
            self.figs_folder.mkdir(parents=True, exist_ok=True)
            figpath = self.figs_folder / f"{filename}.{fig_extension}"
            self.plot_checkpoint(self.pca_data, figpath)
            print(f"Figure saved at {figpath}")
            
    def plot_checkpoint(self, X: np.ndarray, filepath: str):
        """Plot the current space and save it to filepath

        Args:
            X (np.ndarray): the current space, as 3d numpy ndarray
            filepath (str): path to the saved image
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 1])

        plt.savefig(filepath)

        plt.close()
        

    def _compute_error(self, p_idx, visited) -> float:
        """Compute the error for the point p_idx

        Args:
            p_idx (int): index of the point
            visited (list): list of points that were already visited in this step

        Returns:
            float: error for the point p_idx
        """
        w = np.where(np.isin(self.neighbours[p_idx], list(visited)), 10, 1)

        neighbours = self.neighbours[p_idx]
        mcn_indices = self.mcn_index[p_idx].astype(int)

        pn = self.pca_data[p_idx] - self.pca_data[neighbours]
        nm = self.pca_data[mcn_indices] - self.pca_data[neighbours]

        pn_dist = np.linalg.norm(pn, axis=1)
        nm_dist = np.linalg.norm(nm, axis=1)

        cosine = np.sum(pn * nm, axis=1) / (pn_dist * nm_dist)

        cosine = np.clip(cosine, -1, 1)

        angles = np.arccos(cosine)

        err_dist = 0.5 * (pn_dist - self.distances0[p_idx]) / self.avg_dist0
        err_theta = (angles - self.mcn_angles[p_idx]) / np.pi

        total_err = np.sum(w * (err_dist**2 + err_theta**2))
        
        return total_err
    
    def _adjust_point(self, p: int, visited: list[int], s_threshold: int = 30) -> tuple[int, float]:
        """Adjust the point p in the dataset until no improvement is found or the step threshold is reached.

        Args:
            p (int): index of the point to adjust
            visited (list): list of points that were already adjusted in this step
            s_threshold (int, optional): maximum number of steps to adjust the point. Defaults to 30.

        Returns:
            int: steps taken to adjust the point p
            float: error for the point p
        """
        
        lr = self.learning_rate
        improved = True

        err = self._compute_error(p, visited) # Initial error for the point
        s = 0 # Number of steps taken to adjust the point
        while (s < s_threshold) and improved:
            s += 1
            improved = False

            for d in self.d_pres:
                self.pca_data[p,d] += lr
                newerr = self._compute_error(p, visited)

                if newerr >= err:
                    self.pca_data[p,d] -= 2 * lr
                    newerr = self._compute_error(p, visited)
                
                    if newerr >= err:
                        self.pca_data[p, d] += lr
                    else:
                        err = newerr
                        improved = True
                else:
                    err = newerr
                    improved = True
        return s - 1, err
    
    def _step(self) -> float:
        """Manifold sculpting step

        Returns:
            float: mean error for the step
        """
        N = self.pca_data.shape[0]
        
        # Choose a random starting point, initialize the queue and the visited set
        origin = np.random.choice(np.arange(N, dtype=int)) 
        q = deque([origin])
        visited = set()

        # 4.1) Scale down the scaling dimensions
        self.scale_factor *= self.sigma
        self.pca_data[:, self.d_scal] *= self.sigma

        # 4.2) Scale up the preserved dimensions until the average distance is restored
        avg_dist = average_neighbor_distance(self.pca_data, self.neighbours)
        while avg_dist < self.avg_dist0:
            self.pca_data[:, self.d_pres] /= self.sigma
            avg_dist = average_neighbor_distance(self.pca_data, self.neighbours)

        step = 0
        mean_error = 0
        counter = 0
        # Traverse the unvisited points and adjust them,
        # then add their neighbors to the queue and mark them as visited
        while q:
            p_idx = q.popleft() # Return the first element and remove it from the queue
            if p_idx in visited:
                continue
            
            q.extend(self.neighbours[p_idx, :])

            adjusting_steps, err = self._adjust_point(p_idx, visited)
            step += adjusting_steps
            mean_error += err
            counter += 1
            visited.add(p_idx)

        mean_error /= counter
        
        if step < N:
            self.learning_rate *= 0.90
        else:
            self.learning_rate /= 0.90

        return mean_error