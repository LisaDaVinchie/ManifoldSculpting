import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

class ManifoldSculpting:
    def __init__(self, X: np.ndarray, n_neighbors: int, D_pres: int):
        self.X = X
        self.N_points = X.shape[0]
        self.D = X.shape[1]
        self.n_neighbors = n_neighbors
        self.D_pres = D_pres
    
    def find_neighbors(self):
        # Step 1: Find the neighbors of each point
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean').fit(self.X)

        self.distances, self.N_indexes = nbrs.kneighbors(self.X)
    
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
                self.X[i, j] = np.dot(self.X[i, :], G[:, j])
        
        self.X += mean

    def fit_transform(self):
        self.find_neighbors() # Step 1
        self.AlignAxes_PC() # Step 2
        self.m, self.c = FindMCN().most_collinear_ngbr # Step 3
        self.step4b()
        return self.X


class FindMCN(ManifoldSculpting):
    def __init__(self):
        super().__init__()
        self.m = np.zeros((self.N_points, self.n_neighbors), dtype=int)
        self.c = np.zeros((self.N_points, self.n_neighbors), dtype=float)
    
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
        
class AdjustPoints(ManifoldSculpting):
    def __init__(self, cosines: np.ndarray, D_pres: np.ndarray):
        self.sigma: float = 0.99 # scaling factor

        self.cosines = cosines # cosines of the angles between the points and their neighbors
        self.D_scal = self.D - self.D_pres # number of dimensions to scale

        self.dist_avg = np.mean(self.distances[:, 1:]) # average distance to the neighbors
        self.eta = self.dist_avg.copy() # Copy of the average distance to the neighbors
        self.dist_curr = self.distances.copy() # Distances after scaling
        # self.m_curr = np.zeros((self.N_points, self.n_neighbors), dtype=int) # Most collinear neighbor indexes after scaling
        # self.c_curr = np.zeros((self.N_points, self.n_neighbors), dtype=float) # Cosines after scaling
        self.omega = np.ones((self.N_points, self.n_neighbors - 1), dtype=int) # Omega values


    def ComputeError(self, p_cur_idx: np.ndarray):
        """Computes the error value of the current configuration.
            - p_cur_idx: index of the current point
        """

        denom_dist: float = 1 / (2 * self.dist_avg)
        pi_inv: float = 1 / np.pi

        error: float = 0.0

        for j in range(self.n_neighbors):
            dist_term = (self.distances[p_cur_idx, j] - self.dist_curr[p_cur_idx, j]) * denom_dist
            angle_diff = np.arccos(self.cosines[p_cur_idx, j]) - np.arccos(self.c_curr[p_cur_idx, j])

            angle_term = np.max(0, angle_diff) * pi_inv

            error += self.omega[p_cur_idx, j] * (dist_term * dist_term + angle_term * angle_term)

        return error
    
    def AdjustPoints(self, p_cur_idx: np.ndarray):
        s: int = -1 # Initialize the number of steps
        improved: bool = True # Initialize the improved flag

        X_new = self.X.copy() # Copy the dataset

        while improved:
            s += 1 # Increment the number of steps
            improved = False # Reset the improved flag
            error: float = self.ComputeError(p_cur_idx) # Compute the error value

            for d in range(self.D_pres):
                X_new[p_cur_idx, d] += self.eta # Add the eta value to the current point
                self.dist_curr[p_cur_idx, :] = np.linalg.norm(X_new[p_cur_idx, :] - X_new[self.N_indexes[p_cur_idx, :]], axis=1) # Compute the distances
                self.m_curr, self.c_curr = FindMCN(X_new, self.N_indexes, self.dist_curr).most_collinear_ngbr() # Compute the most collinear neighbors

                if self.ComputeError(p_cur_idx) > error:
                    X_new[p_cur_idx, d] -= 2 * self.eta
                    self.dist_curr[p_cur_idx, :] = np.linalg.norm(X_new[p_cur_idx, :] - X_new[self.N_indexes[p_cur_idx, :]], axis=1) # Compute the distances
                    self.m_curr, self.c_curr = FindMCN(X_new, self.N_indexes, self.dist_curr).most_collinear_ngbr() # Compute the most collinear neighbors

                    if self.ComputeError(p_cur_idx) > error:
                        X_new[p_cur_idx, d] += self.eta
                        self.dist_curr[p_cur_idx, :] = np.linalg.norm(X_new[p_cur_idx, :] - X_new[self.N_indexes[p_cur_idx, :]], axis=1) # Compute the distances
                        self.m_curr, self.c_curr = FindMCN(X_new, self.N_indexes, self.dist_curr).most_collinear_ngbr() # Compute the most collinear neighbors
                    else:
                        improved = True
                else:
                    improved = True
        return s
    
    def step4a(self):
        sigma_inv = 1 / self.sigma
        for i in range(self.N_points):
            for j in range(self.D_scal):
                self.X[i, j] *= self.sigma


        while(np.mean(self.distances[:, 1:]) < self.dist_avg):
            for i in range(self.N_points):
                for j in range(self.D_pres):
                    self.X[i, j] *= sigma_inv
        
        model = FindMCN(self.X, self.N_indexes, self.distances)
        self.m_curr, self.c_curr = model.most_collinear_ngbr()

        for i in range(self.N_points):
            for j, n_idx in enumerate(self.N_indexes[i, :]):
                self.dist_curr[i, j] = np.linalg.norm(self.X[i] - self.X[n_idx])
        
    
    def step4b(self):
        steps: int = 0

        X_idxs = np.arange(self.N_points, dtype=int)

        random_idx: int = random.choice(X_idxs)
        
        queue = []
        queue.append(random_idx)

        adj_points = []

        X_idxs = np.delete(X_idxs, random_idx)

        eta_scal: float = 0.9
        eta_scal_inv: float = 1 / eta_scal

        while len(queue) > 0:
            p_cur_idx = queue.pop(0)

            if p_cur_idx in adj_points:
                continue
            else:
                # find omega
                for i in range(1, self.n_neighbors):
                    if self.N_indexes[p_cur_idx, i] in adj_points:
                        self.omega[p_cur_idx, i - 1] = 10
                steps += AdjustPoints(p_cur_idx)
                for i in range(1, self.n_neighbors):
                    queue.append(self.N_indexes[p_cur_idx, i])
                adj_points.append(p_cur_idx)
            
        if steps > self.N_points:
            self.eta *= eta_scal_inv
        else:
            self.eta *= eta_scal