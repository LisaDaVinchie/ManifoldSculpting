import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import random

class ManifoldSculpting:
    def __init__(self, n_neighbors: int = 5, n_components: int = 2, sigma: float = 0.99, iterations: int = 100, rotate: bool = True):
        self.n_neighbors: int = n_neighbors
        self.n_components: int = n_components
        self.scale_factor = 1.0 # learning rate

        self.sigma: float = sigma # scaling factor
        self.sigma_inv: float = 1 / self.sigma # inverse of the scaling factor

        self.rotate: bool = rotate # flag to rotate the dataset
        self.iterations: int = iterations # number of iterations

    def fit_transform(self, X: np.ndarray):
        # Step 1
        self.X = X
        self.N_points: int = self.X.shape[0]
        self.D: int = self.X.shape[1]

        self.distances, self.neighbors = self._findNearestNeighbors(data = self.X, n_neighbors = self.n_neighbors)
        print("Found Neighbors")

        self.dist_avg = self._averageNeighborDistance()

        self.eta = self.dist_avg

        self.mcn_idx, self.mcn_angles = self._mostCollinearNeighbor()
        print("Found Most Collinear Neighbors")


        if self.rotate:
            model = PCA(n_components=self.D)
            self.transformed_X = model.fit_transform(self.X)
            self.D_pres = np.arange(self.n_components)
            self.D_scal = np.arange(self.n_components, self.D)
        else:
            cov = np.cov(self.X.T)
            most_important = np.argsort(-np.diag(cov)).astype(np.int32)
            self.D_pres = most_important[:self.n_components]
            self.D_scal = most_important[self.n_components:]
            self.transformed_X = np.copy(self.X)
        
        print("Rotated")
        
        epoch = 0

        self.best_transformed_X = np.copy(self.transformed_X)

        print("Initialisation")
        while self.scale_factor > 0.01 and epoch < 5:
            mean_error = self._step()
            epoch += 1
            print(f"Epoch: {epoch}, Error: {mean_error}")
        
        
        best_error = np.inf
        diff = np.inf
        errors = []

        print("Starting iterations")

        while(epoch < self.iterations and diff > 0.01):
            old_transformed_X = np.copy(self.transformed_X)
            print("Settes old dataset")
            mean_error = self._step()
            print("Step done")
            errors.append(mean_error)


            if mean_error < best_error:
                best_error = mean_error
                self.best_transformed_X = np.copy(self.transformed_X)

                diff = np.linalg.norm(self.best_transformed_X - old_transformed_X) / np.linalg.norm(old_transformed_X)

            epoch += 1
            
            print(f"Epoch: {epoch}, Error: {mean_error:.4f}, Diff: {diff:.4f}")
        
        print("Finished")
        
        return self.best_transformed_X[:, self.D_pres], errors

    def _findNearestNeighbors(self, data, n_neighbors: int):
        model = NearestNeighbors(n_neighbors=n_neighbors)
        model.fit(data)
        distances, neighbors = model.kneighbors(data)
        return distances[:, 1:], neighbors[:, 1:]
    
    def _mostCollinearNeighbor(self):
        """
        Calculate the most collinear neighbor of each neighbor of each point in the dataset

        Returns:
            -np.ndarray of ints: m, where m[i, j] is the index of the neighbor of the i-th point that is most collinear with the j-th neighbor of the i-th point
            -np.ndarray of floats: c, where c[i, j] is the cosine of the angle between the i-th point and the m[i, j]-th neighbor
        """
        mcn_index = np.ones((self.N_points, self.n_neighbors - 1), dtype=int) * (-2)
        mcn_angles = np.ones((self.N_points, self.n_neighbors - 1), dtype=float) * (-2)
        for i in range(self.N_points):
            p = self.X[i, :]
            for j in range(self.n_neighbors - 1):
                n = self.X[self.neighbors[i, j], :]
                pn = n - p
                n_idx = self.neighbors[i, j]

                min_angle: float = -2
                min_diff: float = 2
                max_idx: int = -2
                for k in range(self.n_neighbors - 1):
                    m_idx = self.neighbors[n_idx, k]
                    m = self.X[m_idx, :]
                    nm = m - n
                    cosine = np.dot(pn, nm) / (np.linalg.norm(pn) * np.linalg.norm(nm))
                    angle = np.arccos(np.minimum(1, np.maximum(-1, np.minimum(1, cosine))))
                    diff = np.abs(np.pi - angle)
                    if diff < min_diff:
                        min_diff = diff
                        min_angle = angle
                        max_idx = m_idx



                mcn_index[i, j] = max_idx
                mcn_angles[i, j] = min_angle
                
        return mcn_index, mcn_angles
    
    def _averageNeighborDistance(self):
        """
        Calculate the average distance between the points and their neighbors

        Returns:
            -float: average distance between the points and their neighbors
        """
        dist_sum: float = 0
        for i in range(self.N_points):
            for j in range(self.n_neighbors - 1):
                dist_sum += np.linalg.norm(self.X[i, :] - self.X[self.neighbors[i, j], :])
        return dist_sum / (self.N_points * (self.n_neighbors - 1))
    
    def _step(self):
        first_point = np.random.randint(0, self.N_points)

        q = []
        q.append(first_point)
        visited = []

        self.scale_factor *= self.sigma

        # Scale down dimensions to scale
        self.transformed_X[:, self.D_scal] *= self.sigma

        while self._averageNeighborDistance() < self.dist_avg:
            self.transformed_X[:, self.D_pres] /= self.sigma
        
        step = 0
        mean_error = 0
        counter = 0
        while q:
            p = q.pop(0)

            if p not in visited:
                for n in self.neighbors[p, :]:
                    q.append(n)
                
                s, error = self._adjustPoints(p, visited, self.eta)
                step += s
                mean_error += error
                visited.append(p)
                counter += 1
        
        mean_error /= counter

        if step > self.N_points:
            self.eta /= 0.9
        else:
            self.eta *= 0.9

        return mean_error

    def _computeError(self, p_cur_idx: int, visited: list) -> float:

        omega = np.zeros(self.n_neighbors - 1)
        for i in range(self.n_neighbors - 1):
            if self.neighbors[p_cur_idx, i] in visited:
                omega[i] = 0
            else:
                omega[i] = 1
        
        error: float = 0.0
        pi_inv = 1 / np.pi
        avg_dist_inv = 1 / (2 * self.dist_avg)


        for j in range(self.n_neighbors - 1):
            n = self.neighbors[p_cur_idx, j]
            c = self.mcn_idx[p_cur_idx, j]

            # calculate new mcn angles

            pn = self.transformed_X[p_cur_idx, :] - self.transformed_X[n, :]
            nm = self.transformed_X[c, :] - self.transformed_X[n, :]
            dist_pn = np.linalg.norm(pn)
            dist_nm = np.linalg.norm(nm)
            cosine = np.dot(pn, nm) / (dist_pn * dist_nm)

            new_mcn_angle = np.arccos(np.minimum(1, np.maximum(-1, cosine)))

            angle_diff = new_mcn_angle - self.mcn_angles[p_cur_idx, j - 1]
            angle_term = np.maximum(0, angle_diff) * pi_inv

            dist_term = (dist_pn - self.distances[p_cur_idx, j]) * avg_dist_inv
            
            error += omega[j] * (dist_term * dist_term + angle_term * angle_term)
        
        return error
    
    def _adjustPoints(self, p_cur_idx: int, visited: list, eta: float):
        s: int = -1 # Initialize the number of steps
        improved: bool = True # Initialize the improved flag

        error = self._computeError(p_cur_idx, visited)

        while improved and s < 30:
            s += 1
            improved = False

            for j in self.D_pres:
                self.transformed_X[p_cur_idx, j] += eta
                new_error = self._computeError(p_cur_idx, visited)

                if new_error >= error:
                    self.transformed_X[p_cur_idx, j] -= 2 * eta
                    new_error = self._computeError(p_cur_idx, visited)

                    if new_error >= error:
                        self.transformed_X[p_cur_idx, j] += eta
                    else:
                        error = new_error
                        improved = True
                else:
                    improved = True

        return s, error