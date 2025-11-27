import numpy as np

def computePCA(data) -> np.ndarray:
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