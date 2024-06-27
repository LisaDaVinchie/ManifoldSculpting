import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

class ManifoldSculpting:
    def __init__(self, n_neighbors: int, D_pres: int):
        self.n_neighbors: int = n_neighbors
        self.D_pres: int = D_pres
        # self.iterations: int = iterations
        self.X: np.ndarray = None

        self.sigma: float = 0.99 # scaling factor
        self.sigma_inv: float = 1 / self.sigma # inverse of the scaling factor
        
        self.c: np.ndarray = None # cosines of the angles between the points and their most collinear neighbors
        self.distances: np.ndarray = None # distances between the points and their neighbors
        self.dist_avg: float = None # average distance between the points and their neighbors
        self.eta: float = None # step size
        self.m: np.ndarray = None # most collinear neighbors of the neighbors of the points
        self.neighbors: np.ndarray = None # neighbors of the points
        self.new_c: np.ndarray = None # new cosines of the angles between the points and their most collinear neighbors
        self.new_dist: np.ndarray = None # new distances between the points and their neighbors
        self.omega: np.ndarray = None # weight of the neighbors of the points
    
    def _calculateCosine(self, p_idx: int, n_idx: int, m_idx: int) -> float:
        """_summary_

        Args:
            p_idx (int): index of the point p
            n_idx (int): index of the point n, neighbor of p
            m_idx (int): index of the point m, neighbor of n

        Returns:
            float: cosine of the angle between the vectors pn and nm
        """
        p = self.X[p_idx, :]
        n = self.X[n_idx, :]
        m = self.X[m_idx, :]

        if np.array_equal(p, m):
            return 0
        else:
            pn = n - p # vector from p to n
            nm = m - n # vector from n to m
            dot_prod = np.dot(pn, nm) # dot product of pn and nm
            norm_prod = np.linalg.norm(pn) * np.linalg.norm(nm) # product of the norms of pn and nm

            cosine = dot_prod / norm_prod # cosine of the angle between pn and nm
        
            if abs(cosine) > 1:
                raise ValueError("Invalid cosine value. Cosine cannot be greater than 1.")
            else:
                return cosine
    
    def _mostCollinearNeighbor(self):
        """
        Calculate the most collinear neighbor of each neighbor of each point in the dataset

        Returns:
            -np.ndarray of ints: m, where m[i, j] is the index of the neighbor of the i-th point that is most collinear with the j-th neighbor of the i-th point
            -np.ndarray of floats: c, where c[i, j] is the cosine of the angle between the i-th point and the m[i, j]-th neighbor
        """
        m = np.ones((self.N_points, self.n_neighbors), dtype=int) * (-2)
        c = np.ones((self.N_points, self.n_neighbors), dtype=float) * (-2)
        for i in range(self.N_points):
            p = self.X[i, :]
            for j in range(1, self.n_neighbors):
                n_idx = self.neighbors[i, j]

                max_cosine: float = -2.0
                max_idx: int = -2
                for k in range(1, self.n_neighbors):
                    m_idx = self.neighbors[n_idx, k]
                    if np.array_equal(p, self.X[m_idx, :]):
                        cosine = 0
                    else:
                        cosine = self._calculateCosine(i, n_idx, m_idx)

                    if cosine > max_cosine:
                        max_cosine = cosine
                        max_idx = m_idx

                m[i, j] = max_idx
                c[i, j] = max_cosine
                
        return m, c
    
    def _alignAxesPC(self):

        """Aligns the axes of the dataset P using the Principal Components method.

        Parameters:
        - P: dataset to be aligned, with shape (N, D), where N is the number of points and D is the number of dimensions
        - Dpres: number of dimensions to be preserved
        """

        # Center the data
        mean: float = np.mean(self.X, axis=0) # Mean for every dimension
        self.X -= mean

        Q = self.X.copy()

        G = np.eye(self.D) # Matrix of standard basis vectors

        # Find principal components
        for k in range(self.D_pres):
            c = np.random.rand(self.D) # Random vector of dimension D

            for _ in range(10):
                t = np.zeros(self.D)

                for q in Q: # For each row of Q
                    t += np.dot(q, c) * q
                c = t / np.linalg.norm(t)

            for q in Q: # For each row of Q
                q -= np.dot(c, q) * c
            
            a = G[:, k]

            b = (c - np.dot(a, c) * a) / np.linalg.norm(c - np.dot(a, c) * a)

            phi  = np.arctan(np.dot(b, c) / np.dot(a, c))

            for j in np.arange(k, self.D):
                u = np.dot(a, G[:, j])
                v = np.dot(b, G[:, j])

                G[:, j] -= u * a
                G[:, j] -= v * b

                r = np.sqrt(u * u + v * v)
                theta = np.arctan(v / u)

                u = r * np.cos(theta + phi)
                v = r * np.sin(theta + phi)

                G[:, j] += u * a
                G[:, j] += v * b
        
        for i in range(self.N_points):
            for j in range(self.D):
                self.X[i, j] = np.dot(self.X[i, :], G[:, j]) + mean[j]
    
    def _updateDistances(self, rows: list = None, cols: list = None):
        if rows is None:
            rows = np.arange(self.N_points)
        if cols is None:
            cols = np.arange(self.n_neighbors)
        for i in rows:
            for j in cols:
                self.new_dist[i, j] = np.linalg.norm(self.X[i] - self.X[self.neighbors[i, j]])

    def _updateCosines(self, rows: list = None, cols: list = None):
        if rows is None:
            rows = np.arange(self.N_points)
        if cols is None:
            cols = np.arange(1, self.n_neighbors)

        for i in rows:
            for j in cols:
                idx_n = self.neighbors[i, j]
                idx_m = self.m[i, j]

                if np.array_equal(self.X[i, :], self.X[idx_m, :]):
                    self.new_c[i, j] = 0
                else:
                    # if idx_m not in self.neighbors[idx_n, :]:
                    #     print("Neighbors of i: " + str(self.neighbors[i, :]))
                    #     print("Neighbors of " + str(idx_n)+ ": " + str(self.neighbors[idx_n, :]))
                    #     print("m: " + str(idx_m))

                    self.new_c[i, j] = self._calculateCosine(i, idx_n, idx_m)

    def _computeError(self, p_cur_idx: int):
        error: float = 0.0
        pi_inv = 1 / np.pi
        avg_dist_inv = 1 / (2 * self.dist_avg)
        # dist_term = (self.new_dist[p_cur_idx, 1:] - self.distances[p_cur_idx, 1:]) * avg_dist_inv
        # angle_diff = np.arccos(self.new_c[p_cur_idx, 1:]) - np.arccos(self.c[p_cur_idx, 1:])
        # angle_term = np.maximum(0, angle_diff) * pi_inv


        for j in range(1, self.n_neighbors):
            dist_term = (self.new_dist[p_cur_idx, j] - self.distances[p_cur_idx, j]) * avg_dist_inv
            angle_diff = np.arccos(self.new_c[p_cur_idx, j]) - np.arccos(self.c[p_cur_idx, j])
            angle_term = np.maximum(0, angle_diff) * pi_inv

            error += self.omega[p_cur_idx, j] * (dist_term * dist_term + angle_term * angle_term)
        
        return error
    
    def _adjustPoints(self, p_cur_idx: int, eta: float):
        s: int = -1 # Initialize the number of steps
        improved: bool = True # Initialize the improved flag

        while improved:
            s += 1
            improved = False
            self.new_c = self.c.copy()
            error = self._computeError(p_cur_idx)

            for j in np.arange(self.D_pres):
                self.X[p_cur_idx, j] += eta
                self._updateDistances(rows=[p_cur_idx], cols = None)
                self._updateCosines(rows=[p_cur_idx], cols = None)

                if self._computeError(p_cur_idx) > error:
                    self.X[p_cur_idx, j] -= 2 * eta
                    self._updateDistances(rows=[p_cur_idx], cols = None)

                    self._updateCosines(rows=[p_cur_idx], cols = None)

                    if self._computeError(p_cur_idx) > error:
                        self.X[p_cur_idx, j] += eta
                        self._updateDistances(rows=[p_cur_idx], cols = None)
                        self._updateCosines(rows=[p_cur_idx], cols = None)
                    else:
                        improved = True
                else:
                    improved = True

        return s
                
    def fit_transform(self, X: np.ndarray):
        # Step 1
        self.X = X
        self.N_points: int = self.X.shape[0]
        self.D: int = self.X.shape[1]
        self.D_scal: int = self.D - self.D_pres
        self.omega = np.ones((self.N_points, self.n_neighbors), dtype=int)

        model = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X)
        self.distances, self.neighbors = model.kneighbors(self.X)
        print("Step 1 done.\n")

        # Step 2
        self.m, self.c = self._mostCollinearNeighbor()

        self.dist_avg = np.mean(self.distances[:, 1:])
        self.eta = self.dist_avg.copy()

        print("Step 2 done.\n")

        # Step 3
        self._alignAxesPC()

        print("Step 3 done.\n")

        # Step 4
        self.new_dist = self.distances.copy()

        sum_changes: float = 0.0



        while sum_changes > 0.1:
            X_old = self.X.copy()
            # Step 4a
            self.X[:, :-self.D_scal] *= self.sigma

            # recalculate distances
            self._updateDistances(rows=None, cols=None)
            
            while np.mean(self.distances[:, 1:]) < self.dist_avg:
                self.X[:, :self.D] *= self.sigma_inv
                self._updateDistances(rows=None, cols=None)
                self.distances = self.new_dist.copy()

            # Step 4b
            queue = []
            random_idx: int = random.choice(self.neighbors[:, 0])
            queue.append(random_idx)

            steps: int = 0
            adjusted = []
            while len(queue) > 0:
                p_cur_idx = queue.pop(0)
                if p_cur_idx not in adjusted:
                    steps += self._adjustPoints(p_cur_idx, self.eta)
                    adjusted.append(p_cur_idx)
                    self.omega[p_cur_idx, :] = 10

                    for j in range(1, self.n_neighbors):
                        if self.neighbors[p_cur_idx, j] not in adjusted:
                            queue.append(int(self.neighbors[p_cur_idx, j]))

            if steps >= self.D:
                self.eta *= 1/0.9
            else:
                self.eta *= 0.9

            sum_changes = np.linalg.norm(X_old - self.X) / (self.N_points * self.D)


        
        print("Step 4 done.\n")
    
        # Step 5

        return self.X[:, :self.D_pres]

       