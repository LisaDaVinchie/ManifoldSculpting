import numpy as np
def swiss_roll(N):
    """Generates a swiss roll dataset with N samples."""
    dataset = np.ndarray((N, 3))

    
    dataset[:, 1] = np.random.uniform(-6, 6, N)

    N_inv = 1 / N

    t = 8 * np.arange(N) * N_inv + 2

    dataset[:, 0] = t * np.sin(t)
    dataset[:, 2] = t * np.cos(t)

    return dataset