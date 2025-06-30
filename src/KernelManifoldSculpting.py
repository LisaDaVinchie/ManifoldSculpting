import numpy as np
from collections import deque
from sklearn.decomposition import KernelPCA

class ManifoldSculpting():

    def __init__(self, n_neighbors: int = 10, n_components: int = 2, iterations: int = 100, sigma: float = 0.99, perform_pca: bool = True, max_iter_no_change: int = 30):
        """Used to pass parameters to che class

        Args:
            n_neighbors (int, optional): Number of neighbors for each point. Defaults to 5.
            n_components (int, optional): Number of dimensions to preserve. Defaults to 2.
            iterations (int, optional): Max number of iterations. Defaults to 100.
            sigma (float, optional): Scale factor. Defaults to 0.99.
            perform_pca (bool, optional): Decide if you want to perform PCA. Defaults to True.
            max_iter_no_change (int, optional): Maximum number of iterations with no change in the error. Defaults to 30.
        """

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.iterations = iterations
        self.sigma = sigma
        self.rotate = perform_pca

        self.scale_factor = 1

        self.max_iter_no_change = max_iter_no_change

    def fit(self, data, save_checkpoints: bool = False, folder: str = '', checkpoint_interval: int = 10):
        """Pass the dataset to transform it into a lower dimension

        Args:
            data (MatrixLike): dataset to transfrom, made as a matrix of shape (n_samples, n_features)
            save_checkpoints (bool, optional): Saves intermediate transformations of the dataset in .npy format. Defaults to False.
            folder (str, optional): Where to save the checkpoints. Defaults to ''.
            checkpoint_interval (int, optional): number of epochs between one checkpoint and another one. Defaults to 10.

        Returns:
            MatrixLike: transformed dataset
        """
        self.data = data
        self.n_points = self.data.shape[0]
        
        self.neighbours, self.distances0, self.avg_dist0= self._findKNN()
        self.mcn_index, self.mcn_angles = self._findMCN(self.data, self.neighbours)
        self.learning_rate = self.avg_dist0

        if self.rotate:
            self.pca_data = self._computePCA()
            self.d_pres = np.arange(self.n_components,dtype=np.int32)
            self.d_scal = np.arange(self.n_components, self.data.shape[1],dtype=np.int32)
        else:
            cov = np.cov(self.data.T)
            most_important = np.argsort(-np.diag(cov)).astype(np.int32)
            self.d_pres = most_important[:self.n_components]
            self.d_scal = most_important[self.n_components:]
            self.pca_data = np.copy(self.data)

        if save_checkpoints:
                np.save(folder + f"checkpoint_0.npy", self.pca_data)

        epoch = 1
        while self.scale_factor > 0.01:
            mean_error = self._step()
            epoch += 1

            if save_checkpoints and epoch % checkpoint_interval == 0:
                np.save(folder + f"checkpoint_{epoch}.npy", self.pca_data)
  
        epochs_since_improvement = 0
        best_error = np.inf
        
        while (epoch < self.iterations) and (epochs_since_improvement < self.max_iter_no_change):
            
            mean_error = self._step()

            if mean_error < best_error:
                best_error = mean_error
                self.best_data = np.copy(self.pca_data)
                self.best_error = best_error
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            epoch += 1
            
            if save_checkpoints and epoch % checkpoint_interval == 0:
                np.save(folder + f"checkpoint_{epoch}.npy", self.pca_data)

        self.elapsed_epochs = epoch
        self.last_error = mean_error

        return self.pca_data

    def _computeError(self, p_idx, visited):
        """Compute the error for the point p_idx

        Args:
            p_idx (int): index of the point
            visited (list): list of points that were already visited in this step

        Returns:
            float: error for the point p_idx
        """
        w = np.where(np.isin(self.neighbours[p_idx], list(visited)), 10, 1)

        neighbours = self.neighbours[p_idx]
        mcn_indices = self.mcn_index[p_idx].astype(int)

        pn = self.pca_data[p_idx] - self.pca_data[neighbours]
        nm = self.pca_data[mcn_indices] - self.pca_data[neighbours]

        pn_dist = np.linalg.norm(pn, axis=1)
        nm_dist = np.linalg.norm(nm, axis=1)

        cosine = np.sum(pn * nm, axis=1) / (pn_dist * nm_dist)

        cosine = np.clip(cosine, -1, 1)

        angles = np.arccos(cosine)

        err_dist = 0.5 * (pn_dist - self.distances0[p_idx]) / self.avg_dist0
        err_theta = (angles - self.mcn_angles[p_idx]) / np.pi

        total_err = np.sum(w * (err_dist**2 + err_theta**2))
        
        return total_err

    def _findKNN(self):
        """Calculate the K nearest neighbors for each point in the dataset and their distances from the point

        Returns:
            MatrixLike: indexes of the K nearest neighbors for each point
            MatrixLike: distances of the K nearest neighbors for each point
            float: average distance between neighbors
        """
        N = self.data.shape[0]
       
        x2 = np.sum(self.data*self.data,axis = 1)
        data_t = np.copy(self.data.T)
        xx = self.data@data_t
        dist = np.sqrt(np.abs(x2.reshape((-1,1))-2*xx+x2))

        _neigh = np.zeros((N,self.n_neighbors),dtype=np.int32)
        _dist = np.zeros((N,self.n_neighbors),dtype=np.float32)
        for i in range(N):
            _neigh[i] = np.argsort(dist[i])[1:self.n_neighbors+1]
            _dist[i,:] = dist[i,_neigh[i]]
        _ave_dist = np.mean(_dist)

        return _neigh, _dist, _ave_dist
    
    def _findMCN(self, data, neighbors):
        """Find most collinear neighbors for each point in the dataset

        Args:
            data (MatrixLike): datset of shape (n_samples, n_features)
            neighbors (MatrixLike): matrix of indexes of the K nearest neighbors for each point

        Returns:
            MatrixLike: indexes of the most collinear neighbors for each point
            MatrixLike: angles between the point and the most collinear neighbors
        """
        N = data.shape[0]
        mcn_idx = np.zeros((N,self.n_neighbors),dtype=np.int32)
        mcn_angle = np.zeros((N,self.n_neighbors),dtype=np.float32)
        
        for i in range(N):

            p = self.data[i, :]

            for j, n_idx in enumerate(neighbors[i]):
                n = self.data[n_idx, :]
                pn = p - n
                pn_dist = np.linalg.norm(pn)

                nm = self.data[neighbors[n_idx]] - n
                nm_dist = np.linalg.norm(nm, axis=1)

                cosines = np.sum(pn * nm, axis=1) / (pn_dist * nm_dist)
                cosines = np.clip(cosines, -1, 1)

                angles = np.arccos(cosines)

                index = np.argmin(np.abs(angles-np.pi))

                mcn_idx[i,j] = neighbors[n_idx,index]
                mcn_angle[i,j] = angles[index]

        return mcn_idx, mcn_angle    
    
    def _computePCA(self):
        """Compute the kernel PCA of the dataset

        Returns:
            MatrixLike: dataset transformed with PCA
        """
        pca = KernelPCA(n_components=3, kernel='rbf', gamma=0.1)
        return pca.fit_transform(self.data)
    
    def _averageNeighborDistance(self):
        """Computes the average distance between each point and its neighbors

        Returns:
            float: average distance between each point and its neighbors
        """
        dist = 0
        count = 0
        for p_idx in range(self.n_points):
            p = self.pca_data[p_idx]
            for n in self.neighbours[p_idx]:
                count += 1
                dist += np.linalg.norm(p-self.pca_data[n])
        dist /= count
        return dist
    
    def _adjustPoint(self, p, visited):
        """Adjust the point p in the dataset

        Args:
            p (int): index of the point to adjust
            visited (list): list of points that were already adjusted in this step

        Returns:
            int: steps taken to adjust the point p
            float: error for the point p
        """
        lr = self.learning_rate
        improved = True

        err = self._computeError(p,visited)
        s = 0
        while (s<30) and improved:
            s+=1
            improved = False

            for d in self.d_pres:
                self.pca_data[p,d] += lr
                newerr = self._computeError(p,visited)

                if newerr >= err:
                    self.pca_data[p,d] -= 2*lr
                    newerr = self._computeError(p,visited)
                
                    if newerr >= err:
                        self.pca_data[p,d] += lr
                    else:
                        err = newerr
                        improved = True
                else:
                    err = newerr
                    improved = True
        return s-1, err
    
    def _step(self):
        """Manifold sculpting step

        Returns:
            float: mean error for the step
        """
        N = self.pca_data.shape[0]
        
        origin = np.random.choice(np.arange(N, dtype=int))

        q = deque([origin])
        visited = set()

        self.scale_factor *= self.sigma
       
        self.pca_data[:,self.d_scal] *= self.sigma
        
        while self._averageNeighborDistance() < self.avg_dist0:
            self.pca_data[:,self.d_pres] /= self.sigma


        step = 0
        mean_error = 0
        counter = 0
        while q:
            p_idx = q.popleft()
            if p_idx in visited:
                continue
            
            q.extend(self.neighbours[p_idx, :])

            s,err = self._adjustPoint(p_idx,visited)
            step += s
            mean_error += err
            counter += 1
            visited.add(p_idx)

        mean_error /= counter

        if step < N:
            self.learning_rate *= 0.90
        else:
            self.learning_rate /= 0.90

        return mean_error