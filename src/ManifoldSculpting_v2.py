import numpy as np
from collections import deque

class ManifoldSculpting():
    # general note: might want to pass the data to all functions
    # instead of using self.pca_data
    def __init__(self, n_neighbors=5, n_components = 2, iterations = 100, sigma = 0.99, rotate = True, max_iter_no_change = 30):

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.iterations = iterations
        self.sigma = sigma
        self.rotate = rotate

        self.scale_factor = 1

        self.max_iter_no_change = max_iter_no_change

    def fit(self, data):
        self.data = data
        self.n_points = self.data.shape[0]
        # find neighbours, distances and angles
        self.neighbours, self.distances0, self.avg_dist0= self._findKNN()
        self.mcn_index, self.mcn_angles = self._findMCN(self.data, self.neighbours)
        self.learning_rate = self.avg_dist0

        print('neighbours found')

        # PCA step
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

        print('PCA done')


        # adjust variables for a bunch of times to get to a
        # reasonable point to start comparing errors
        print('Starting initialisation')
        epoch = 1
        while self.scale_factor > 0.01:
            mean_error = self._step()
            epoch += 1

            if epoch % 10 == 0:
                print(f"\tEpoch {epoch}, error {mean_error:.4f}, lr {self.learning_rate}")

            
        
        print('Initialisation done, epochs:', epoch)

        
        epochs_since_improvement = 0
        best_error = np.inf
        print('Starting iterations')
        # adjust variables until error does not change or reached max iterations
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
            
            if epoch % 10 == 0 or epochs_since_improvement == self.max_iter_no_change:
                print(f"\tEpoch {epoch}, error {mean_error:.4f}")

        self.elapsed_epochs = epoch
        self.last_error = mean_error

        return self.pca_data

    def _computeError(self, p_idx, visited):

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

        # # iterate over all neighbours and compute distances and angles
        # total_err = 0
        # for i in range(self.n_neighbors):
        #     n = self.neighbours[p_idx,i]
        #     c = int(self.mcn_index[p_idx,i])
        
        #     a = self.pca_data[p_idx] - self.pca_data[n]
        #     b = self.pca_data[c] - self.pca_data[n]
        #     la = np.linalg.norm(a)
        #     lb = np.linalg.norm(b)
        #     theta = np.arccos(np.minimum(1,np.maximum(np.dot(a,b)/(la*lb),-1)))

        #     err_dist = 0.5*(la-self.distances0[p_idx,i])/self.avg_dist0
        #     err_theta = (theta-self.mcn_angles[p_idx,i])/np.pi
        #     total_err += w[i] * (err_dist*err_dist + err_theta*err_theta)
        
        return total_err

    def _findKNN(self):
        """find neighbours, distances, average distance, colinear points, angle 
        for the data on which MnaifoldSculpting is being trained.

        Returns
        -------
        tuple of numpy.ndarray
            (neighbours, distances, average distance, colinear points, angle)
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
        N = data.shape[0]
        # find the most colinear point for each point-neighbour pair
        mcn_idx = np.zeros((N,self.n_neighbors),dtype=np.int32)
        mcn_angle = np.zeros((N,self.n_neighbors),dtype=np.float32)

        
        # iterate over all points
        for i in range(N):
            # for each point iterate over neighbours

            p = self.data[i, :]

            for j, n_idx in enumerate(neighbors[i]):
                n = self.data[n_idx, :]
                pn = p - n
                pn_dist = np.linalg.norm(pn)

                # compute and keep track of all angles between i-j-k, where
                # k are neighbours of j

                nm = self.data[neighbors[n_idx]] - n
                nm_dist = np.linalg.norm(nm, axis=1)

                cosines = np.sum(pn * nm, axis=1) / (pn_dist * nm_dist)
                cosines = np.clip(cosines, -1, 1)

                angles = np.arccos(cosines)

                # choose the point such that the angle is the closest to pi
                index = np.argmin(np.abs(angles-np.pi))

                mcn_idx[i,j] = neighbors[n_idx,index]
                mcn_angle[i,j] = angles[index]

        return mcn_idx, mcn_angle    
    
    def _computePCA(self):
        """perform PCA on data to which ManifoldSculpting is being fit.

        Parameters
        ----------
        data : numpy.ndarray
            array of shape [npoints, nfeatures]

        Returns
        -------
        numpy.ndarray
            data alligned to principal components
        """
        cov = np.cov(self.data.T)
        eigval,eig = np.linalg.eig(cov)
        index = np.argsort(-eigval)
        eigvec = np.copy(eig[:,index].astype(np.float32))

        return self.data@eigvec
    
    def _averageNeighborDistance(self):
        """compute average distance between neighbours

        Returns
        -------
        float
            average neighbour distance
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
        lr = self.learning_rate
        #* np.random.uniform(0.6,1)
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
        counter = 0 # should be the number of points at the end of loop

        # iterate over items in queue
        while q:
            # get the next point in queue and skip it if already adjusted
            p_idx = q.popleft()
            if p_idx in visited:
                continue

            # print(f'\rit:{iter}, p:{p}')
            # enqueue all the point's neighbours
            
            q.extend(self.neighbours[p_idx, :])

            s,err = self._adjustPoint(p_idx,visited)
            step += s
            mean_error += err
            counter += 1
            visited.add(p_idx)

        mean_error /= counter

        # if not many improvements, reduce lr.
        # if many steps, increase lr.
        # note: values are from the authors' implementation, no clue why they are like this
        if step < N:
            # self.learning_rate *= 0.87
            self.learning_rate *= 0.90
        else:
            # self.learning_rate /= 0.91
            self.learning_rate /= 0.90

        return mean_error