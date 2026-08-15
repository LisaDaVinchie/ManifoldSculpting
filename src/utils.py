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

def find_knn(data: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray, float]:
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

def find_mcn(
    data: np.ndarray,
    neighbors: np.ndarray,
    n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find most collinear neighbors for each point in the dataset

    Args:
        data (np.ndarray): dataset, as a (N, 3) ndarray of 3D points
        neighbors (np.ndarray): neighbors matrix, as a (N, K) ndarray of the indices of the
            K nearest neighbors for each point
        n_neighbors (int): The number of neighbors to calculate

    Returns:
        tuple[np.ndarray, np.ndarray]: The indices of the most collinear neighbors for each point
        and the angles to the most collinear neighbors for each point
    """
    n_points = data.shape[0]
    mcn_idx = np.zeros((n_points, n_neighbors), dtype=np.int32)
    mcn_angle = np.zeros((n_points, n_neighbors), dtype=np.float32)

    for i in range(n_points):

        p = data[i, :]

        for j, n_idx in enumerate(neighbors[i]):
            angle, neighbors = _calculate_point_mcn(data, neighbors, p, n_idx)

            mcn_idx[i,j] = neighbors
            mcn_angle[i,j] = angle

    return mcn_idx, mcn_angle

def _calculate_point_mcn(
    data: np.ndarray,
    neighbors: np.ndarray,
    p: np.ndarray,
    n_idx: int
) -> tuple[float, float]:
    """Find the `p`'s and `n`'s most collinear neighbor

    Args:
        data (np.ndarray): dataset, as a (N, 3) numpy ndarray
        neighbors (np.ndarray): neighbors matrix, as a (N, 3) numpy ndarray
        p (np.ndarray): point, as a (1, 3)
        n_idx (int): index of `p`'s neighbor `n`

    Returns:
        tuple[float, float]: angle and index of `p`'s and `n`'s most collinear neighbor.  
    """
    n = data[n_idx, :]
    pn = p - n
    pn_dist = np.linalg.norm(pn)

    nm = data[neighbors[n_idx]] - n
    nm_dist = np.linalg.norm(nm, axis=1)

    cosines = np.sum(pn * nm, axis=1) / (pn_dist * nm_dist)
    cosines = np.clip(cosines, -1, 1)

    angles = np.arccos(cosines)

    index = np.argmin(np.abs(angles - np.pi))

    return angles[index], neighbors[n_idx, index]

def average_neighbor_distance(data: np.ndarray, neighbors: np.ndarray) -> float:
    """Computes the average distance between each point and its neighbors.
    Args:
        data (np.ndarray): The dataset
        neighbors (np.ndarray): The neighbors for each point
        
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

    n_points = data.shape[0]

    x2 = np.sum(data * data, axis = 1)
    data_t = np.copy(data.T)
    xx = data @ data_t
    dist = np.sqrt(np.abs(x2.reshape((-1,1)) - 2 * xx + x2))
    dist = np.mean(dist[np.arange(n_points)[:,None], neighbors])

    return dist
