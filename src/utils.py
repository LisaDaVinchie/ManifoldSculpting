"""Manifold sculpting helper functions"""

import numpy as np

def compute_pca(data: np.ndarray) -> np.ndarray:
    """Compute the kernel PCA of the dataset
    Args:
        data (np.ndarray): The dataset to compute the PCA on
    Returns:
        np.ndarray: The PCA of the dataset
    """
    cov = np.cov(data.T)
    eigval,eig = np.linalg.eig(cov)
    index = np.argsort(-eigval)
    eigvec = np.copy(eig[:,index].astype(np.float32))

    return data@eigvec

def find_KNN(data: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate the K nearest neighbors for each point in
    the dataset and their distances from the point.

    Args:
        data (np.ndarray): The dataset to calculate the KNN on
        n_neighbors (int): The number of neighbors to calculate
        
    Returns:
        tuple[np.ndarray, np.ndarray, float]: The indices of the K nearest neighbors for each point,
        the distances to the K nearest neighbors for each point and
        the average distance to the K nearest neighbors across all points
    """
    n_points = data.shape[0]

    x2 = np.sum(data * data, axis = 1)
    data_t = np.copy(data.T)
    xx = data @ data_t
    dist = np.sqrt(np.abs(x2.reshape((-1,1)) - 2 * xx + x2))

    neighbors = np.zeros((n_points, n_neighbors), dtype=np.int32)
    distances = np.zeros((n_points, n_neighbors), dtype=np.float32)
    for i in range(n_points):
        neighbors[i] = np.argsort(dist[i])[1 : n_neighbors + 1]
        distances[i, :] = dist[i, neighbors[i]]

    average_dist = np.mean(distances)

    return neighbors, distances, average_dist

def find_MCN(data: np.ndarray, neighbors: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Find most collinear neighbors for each point in the dataset

    Args:
        data (np.ndarray): The dataset to calculate the MCN on
        neighbors (np.ndarray): The K nearest neighbors for each point
        n_neighbors (int): The number of neighbors to calculate

    Returns:
        tuple[np.ndarray, np.ndarray]: The indices of the most collinear neighbors for each point
        and the angles to the most collinear neighbors for each point
    """
    N = data.shape[0]
    mcn_idx = np.zeros((N, n_neighbors), dtype=np.int32)
    mcn_angle = np.zeros((N, n_neighbors), dtype=np.float32)
    
    for i in range(N):

        p = data[i, :]

        for j, n_idx in enumerate(neighbors[i]):
            n = data[n_idx, :]
            pn = p - n
            pn_dist = np.linalg.norm(pn)

            nm = data[neighbors[n_idx]] - n
            nm_dist = np.linalg.norm(nm, axis=1)

            cosines = np.sum(pn * nm, axis=1) / (pn_dist * nm_dist)
            cosines = np.clip(cosines, -1, 1)

            angles = np.arccos(cosines)

            index = np.argmin(np.abs(angles - np.pi))

            mcn_idx[i,j] = neighbors[n_idx,index]
            mcn_angle[i,j] = angles[index]

    return mcn_idx, mcn_angle

def average_neighbor_distance(data: np.ndarray, neighbors: np.ndarray, n_points: int) -> float:
    """Computes the average distance between each point and its neighbors.
    Args:
        data (np.ndarray): The dataset
        neighbors (np.ndarray): The neighbors for each point
        n_points (int): The number of points in the dataset
        
    Returns:
        float: average distance between each point and its neighbors
    """
    
    # dist = 0
    # count = 0
    # for p_idx in range(n_points):
    #     p = data[p_idx]
    #     for n in neighbors[p_idx]:
    #         count += 1
    #         dist += np.linalg.norm(p - data[n])
    # dist /= count
    
    N = data.shape[0]
    
    x2 = np.sum(data * data, axis = 1)
    data_t = np.copy(data.T)
    xx = data @ data_t
    dist = np.sqrt(np.abs(x2.reshape((-1,1)) - 2 * xx + x2))
    dist = np.mean(dist[np.arange(N)[:,None], neighbors])
    
    return dist