import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

class ManifoldSculpting:
    def __init__(self, X: np.ndarray, n_neighbors: int, D_pres: int, iterations: int):
        self.X = X
        self.N_points: int = X.shape[0]
        self.D: int = X.shape[1]
        self.n_neighbors: int = n_neighbors
        self.D_pres: int = D_pres
        self.D_scal: int = self.D - D_pres
        self.iterations: int = iterations

        self.sigma: float = 0.99 # scaling factor
        self.sigma_inv: float = 1 / self.sigma # inverse of the scaling factor

        self.omega = np.ones((self.N_points, self.n_neighbors), dtype=int)
    
    def calculate_cosine(self, p_idx: int, n_idx: int, m_idx: int):
        p: float = self.X[p_idx, :]
        n: float = self.X[n_idx, :]
        m: float= self.X[m_idx, :]
        n_wrt_p: int = np.where(self.N_indexes[p_idx, :] == n_idx)[0][0]
        m_wrt_n: int = np.where(self.N_indexes[n_idx, :] == m_idx)[0][0]

        dist_pn: float = self.distances[p_idx, n_wrt_p]
        dist_nm: float = self.distances[n_idx, m_wrt_n]

        pn = n - p # vector from p to n
        nm = m - n # vector from n to m
        dot_prod: float = np.dot(pn, nm) # dot product of pn and nm
        norm_prod: float = dist_pn * dist_nm # product of the norms of pn and nm

        return dot_prod / norm_prod # cosine of the angle between pn and nm
    
    def most_collinear_ngbr(self):
        m = np.zeros((self.N_points, self.n_neighbors), dtype=int)
        c = np.zeros((self.N_points, self.n_neighbors), dtype=float)
        for i in range(self.N_points):
            p = self.X[i]
            for j in range(1, self.n_neighbors):
                n_idx = self.N_indexes[i, j]

                cosines = np.zeros(self.n_neighbors - 1)

                for k in range(1, self.n_neighbors):
                    if np.array_equal(p, self.X[self.N_indexes[n_idx, k]]):
                        cosines[k - 1] = 0
                    else:
                        cosines[k - 1] = self.calculate_cosine(i, n_idx, self.N_indexes[n_idx, k])
                
                m[i, j] = np.argmax(cosines)
                c[i, j] = cosines[m[i, j]]
                
        return m, c
    
    def AlignAxes_PC(self):

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
                self.X[i, j] = np.dot(self.X[i, :], G[:, j]) + np.mean(self.X[:, j])
    
    def update_distances(self):
        for i in range(self.N_points):
            for j in range(self.n_neighbors):
                self.distances[i, j] = np.linalg.norm(self.X[i] - self.X[self.neighbors[i, j]])
    
    def ComputeError(self, p_cur_idx: int):
        error: float = 0.0
        pi_inv = 1 / np.pi
        avg_dist_inv = 1 / (2 * self.dist_avg)

        dist_term = (self.new_dist[p_cur_idx, :] - self.distances[p_cur_idx, :]) * avg_dist_inv
        angle_diff = np.arccos(self.new_c[p_cur_idx, :] * self.c[p_cur_idx, :])
        angle_term = np.max(0, angle_diff) * pi_inv

        for j in range(self.n_neighbors):
            error += self.omega[p_cur_idx, j] * (dist_term[j] * dist_term[j] + angle_term[j] * angle_term[j])
        
        return error
    
    def AdjustPoints(self, p_cur_idx: int, eta: float):
        s: int = -1 # Initialize the number of steps
        improved: bool = True # Initialize the improved flag

        while improved:
            s += 1
            improved = False
            error = self.ComputeError(p_cur_idx)

            for j in np.arange(self.D_pres):
                self.X[p_cur_idx, j] += eta
                self.update_distances()
                self.new_m, self.new_c = self.most_collinear_ngbr()

                if self.ComputeError(p_cur_idx) > error:
                    self.X[p_cur_idx, j] -= 2 * eta
                    self.update_distances()
                    self.new_m, self.new_c = self.most_collinear_ngbr()

                    if self.ComputeError(p_cur_idx) > error:
                        self.X[p_cur_idx, j] += eta
                        self.update_distances()
                        self.new_m, self.new_c = self.most_collinear_ngbr()
                    else:
                        improved = True
                else:
                    improved = True
                
    def fit_transform(self):
        # Step 1
        model = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X)
        self.neighbors, self.distances = model.kneighbors(self.X)

        # Step 2
        self.m, self.c = self.most_collinear_ngbr()

        self.dist_avg = np.mean(self.distances[:, 1:])
        self.eta = self.dist_avg.copy()

        # Step 3
        self.AlignAxes_PC()

        # Step 4
        for _ in range(self.iterations):
            # Step 4a
            self.X[:, :-self.D_scal] *= self.sigma

            # recalculate distances
            self.update_distances()
            
            while np.mean(self.distances[:, 1:]) < self.dist_avg:
                self.X[:, :self.D] *= self.sigma_inv
                self.update_distances()

            # Step 4b
            queue = []
            random_idx: int = random.choice(self.neighbors[:, 0])
            queue.append(random_idx)

            steps: int = 0
            adjusted = []
            while len(queue) > 0:
                p_cur_idx = queue.pop(0)

                if p_cur_idx not in adjusted:
                    steps += self.AdjustPoints(p_cur_idx, self.eta)
                    adjusted.append(p_cur_idx)
                    queue.append(self.neighbors[p_cur_idx, 1:])
            if steps >= self.D:
                self.eta *= 1/0.9
            else:
                self.eta *= 0.9
    
        # Step 5

        return self.X[:, :self.D_pres]

       