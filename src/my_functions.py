import numpy as np

def swiss_roll(N):
    """Generates a swiss roll dataset with N samples."""
    dataset = np.ndarray((N, 3))
    phi = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, N)

    dataset[:, 0] = phi * np.cos(phi)
    dataset[:, 1] = phi * np.sin(phi)
    dataset[:, 2] = np.random.uniform(0, 10, N)

    return dataset

def calculate_angle(p, n, m, dist_pn, dist_nm):
    """Calculates the angle between the vectors p-n and p-m.
        - p: coordinates of the point p
        - n: coordinates of the point n
        - m: coordinates of the point m
        - dist_pn: distance between p and n
        - dist_nm: distance between p and m
    """
    pn = n - p
    nm = m - p
    dot_prod = np.dot(pn, nm)
    norm_pn = dist_pn * dist_nm
    return np.arccos(dot_prod / norm_pn)