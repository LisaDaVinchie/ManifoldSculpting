import numpy as np
import random
class FindMCN:
    def __init__(self, X: np.ndarray, N_indexes: np.ndarray, distances: np.ndarray):
        self.X = X
        self.N_indexes = N_indexes
        self.distances = distances
        self.N_points = X.shape[0]
        self.n_neighbors = N_indexes.shape[1]
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
        
    


class AdjustPoints:
    def __init__(self, X: np.ndarray, N_indexes: np.ndarray, distances: np.ndarray, angles: np.ndarray, D_pres: np.ndarray, eta: float):
        self.X = X
        self.N_indexes = N_indexes
        self.distances = distances
        self.angles = angles
        self.N_points = X.shape[0]
        self.n_neighbors = N_indexes.shape[1]
        self.dist_avg = np.mean(distances[:, 1:])
        self.eta = self.dist_avg.copy()

    def ComputeError(self, p_cur: np.ndarray):
        """Computes the error value of the current configuration.
            - p_cur: index of the current point
            - distances_curr: distances between p_curr and its neighbors, after transformation, dim: (n_neighbors,)
            - angles_curr: angles between the vectors p-n and p-m, after transformation, dim: (n_neighbors,)
            - distances_init: distances between p_curr and its neighbors, before transformation, dim: (n_neighbors,)
            - angles_init: angles between the vectors p-n and p-m, before transformation, dim: (n_neighbors,)
            - omega: weights of the linear combination, for each neighbor
        """
        
        avg_dist: float = np.mean(self.distances)

        denom_dist: float = 1 / (2 * avg_dist)
        pi_inv: float = 1 / np.pi

        error: float = 0.0

        for j in range(self.n_neighbors):
            dist_term = (self.distances[j] - distances_curr[j]) * denom_dist
            angle_diff = self.angles[j] - angles_curr[j]

            angle_term = np.max(0, angle_diff) * pi_inv

            error += omega[j] * (dist_term * dist_term + angle_term * angle_term)

        return error
    
    def AdjustPoints(self, p_cur: np.ndarray, eta: float, D_pres: np.ndarray, distances_curr: float, angles_curr: float, distances_init: float, angles_init: float, omega: np.ndarray):
        s: int = -1
        improved: bool = True

        while improved:
            s += 1
            improved = False
            error: float = self.ComputeError(p_cur)

            for d in D_pres:
                p_cur[d] += eta

                

                if self.ComputeError(p_cur) > error:
                    p_cur[d] -= 2 * eta

                    if self.ComputeError(p_cur) > error:
                        p_cur[d] += eta
                    else:
                        improved = True
                else:
                    improved = True
        return s
    
    def step4b(self):
        steps: int = 0

        queue = []

        X_idxs = np.arange(self.N_points, dtype=int)

        random_idx: int = random.choice(X_idxs)

        queue.append(random_idx)

        adj_points = []

        X_idxs = np.delete(X_idxs, random_idx)

        eta_scal: float = 0.9

        eta_scal_inv: float = 1 / eta_scal

        while len(queue) > 0:
            p_cur = queue.pop(0)

            if p_cur in adj_points:
                continue
            else:
                # find omega
                omega = np.ones(self.n_neighbors - 1)
                for i in range(1, self.n_neighbors):
                    if self.N_indexes[p_cur, i] in adj_points:
                        omega[i - 1] = 10
                steps += AdjustPoints()
                for i in range(1, self.n_neighbors):
                    queue.append(m[p_cur, i])
                adj_points.append(p_cur)
            
        if steps > self.N_points:
            eta *= eta_scal_inv
        else:
            eta *= eta_scal
        
        return eta



# def calculate_angle(p, n, m, dist_pn, dist_nm):
#     """Calculates the angle between the vectors p-n and p-m.
#         - p: coordinates of the point p
#         - n: coordinates of the point n
#         - m: coordinates of the point m
#         - dist_pn: distance between p and n
#         - dist_nm: distance between p and m
#     """
#     pn = n - p
#     nm = m - p
#     dot_prod = np.dot(pn, nm)
#     norm_pn = dist_pn * dist_nm
#     return np.arccos(dot_prod / norm_pn)

# def calculate_cosine(p: np.ndarray, n: np.ndarray, m: np.ndarray, dist_pn: float, dist_nm: float):
#     """Calculates the angle between the vectors p-n and p-m.
#         - p: coordinates of the point p
#         - n: coordinates of the point n
#         - m: coordinates of the point m
#         - dist_pn: distance between p and n
#         - dist_nm: distance between p and m
#     """
#     pn = n - p # vector from p to n
#     nm = m - n # vector from n to m
#     dot_prod = np.dot(pn, nm) # dot product of pn and nm
#     norm_prod = dist_pn * dist_nm # product of the norms of pn and nm

#     return dot_prod / norm_prod # cosine of the angle between pn and nm

# def most_collinear_ngbr(X: np.ndarray, N_indexes: np.ndarray, distances: np.ndarray):
#     N_points: int = X.shape[0]
#     n_neighbors: int = N_indexes.shape[1]

#     m = np.zeros((N_points, n_neighbors), dtype=int)
#     c = np.zeros((N_points, n_neighbors), dtype=float)
#     for i in range(N_points):
#         p = X[i]
#         for j in range(1, n_neighbors):
#             n_idx = N_indexes[i, j]
#             n = X[N_indexes[i, j]]

#             cosines = np.zeros(n_neighbors - 1)

#             for k in range(1, n_neighbors):
#                 if np.array_equal(p, X[N_indexes[n_idx, k]]):
#                     cosines[k - 1] = 0
#                 else:
#                     cosines[k - 1] = calculate_cosine(p, X[N_indexes[n_idx, k]], n, distances[i, j], distances[n_idx, k])
            
#             m[i, j] = np.argmax(cosines)
#             c[i, j] = cosines[m[i, j]]
            
#     return m, c