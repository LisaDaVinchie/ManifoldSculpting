import unittest
import ManifoldSculpting_v1 as MS
import numpy as np

class testManifoldSculpting(unittest.TestCase):

    def setUp(self) -> None:
        self.X = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4]
        ])
        self.neighbors = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 1, 3, 0, 4],
            [3, 2, 4, 1, 0],
            [4, 3, 2, 1, 0]
        ])

        self.n_neighbors: int = 5

    
    def testFindNeighbors(self):
        distances, neighbors = MS.ManifoldSculpting._findNearestNeighbors(data = self.X, n_neighbors = self.n_neighbors)
        np.testing.assert_array_equal(neighbors, self.neighbors[:, 1:])

    def testMostCollinearNeighbor(self):
        model = MS.ManifoldSculpting()
        model.X = self.X
        model.N_points = self.X.shape[0]
        model.n_neighbors = self.n_neighbors
        model.neighbors = self.neighbors[:, 1:]

        mcn_idx, mcn_angles = model._mostCollinearNeighbor()

        # Just test that all the spaces are filled
        self.assertEqual(np.where(mcn_idx == -2)[0].shape[0], 0)
        self.assertEqual(np.where(mcn_angles == -2)[0].shape[0], 0)
    

if __name__ == '__main__':
    unittest.main()