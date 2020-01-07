import numpy as np
from sklearn.neighbors import NearestNeighbors

'''
FLAME - Fuzzy clustering by Local Approximation of MEmbership

Parameters:
- metric: metric to be used for k nearest neighbor algorithm, supported: manhattan, euclidean (default)
- k_neighbors: number of nearest neighbors to consider when approximating cluster membership (default: 20)
- max_iterations: (optional) maximum number of iterations when approximating cluster membership
- converge_threshold: when the difference between two subsequent iterations reaches this threshold,
                        it is considered to be converged and the algorithm terminates
- min_member_threshold: minimum degree of membership to a cluster in order to be assigned to it 
                        when doing multiple cluster assignment (default: 0.3)
- logging: print information during the clustering process (default: False)                        
'''


class FLAME:
    def __init__(self, metric='euclidean', k_neighbors=20, max_iterations=None,
                 converge_threshold=1e-5, min_member_threshold=0.3, logging=False):
        self.metric = metric
        self.k_neighbors = k_neighbors
        self.n_samples = 0
        self.csos = None
        self.outliers = None
        self.num_clusters = 0
        self.max_iterations = max_iterations
        self.converge_threshold = converge_threshold
        self.min_member_threshold = min_member_threshold
        self.fixed = {}
        self.fuzzy_memberships = None
        self.single_memberships = None
        self.multiple_memberships = None
        self.logging = logging

    def _print(self, p):
        if self.logging:
            print(p)

    def _find_knearest(self, X):
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric=self.metric).fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = distances.T[1:].T  # kneigbors also includes the object itself with distance 0,
        indices = indices.T[1:].T  # we need to remove that
        return distances, indices

    def _compute_densities(self, knn_distances):
        nearest_dist_sums = np.sum(knn_distances, axis=1)
        max_distance = np.max(nearest_dist_sums)
        densities = np.divide(max_distance, nearest_dist_sums)
        return densities

    def _compute_weight_vector(self, knn_distances):
        denom = 0.5 * self.k_neighbors * (self.k_neighbors + 1)
        # Weight depends only on the ranking of distances of the neighbors
        weight_vector = np.array([(self.k_neighbors - j) / denom for j in range(self.k_neighbors)])
        return weight_vector

    def _define_object_types(self, densities, knn_indices):
        csos, outliers, others = [], [], []
        outlier_threshold = np.mean(densities) - 2 * np.std(densities)  # Threshold obtained from FLAME paper
        for i, d in enumerate(densities):
            if np.max(densities[knn_indices[i]]) < d:
                csos.append(i)  # Cluster Supporting Object
                self.fixed[i] = 1
            elif np.min(densities[knn_indices[i]]) > d + outlier_threshold:
                outliers.append(i)  # Outlier
                self.fixed[i] = 1
            else:
                others.append(i)
        return csos, outliers, others

    def _initialize_memberships(self, objects):
        csos, outliers, others = objects
        M = len(csos) + 1  # +1 for the outlier 'cluster'
        self.num_clusters = M
        memberships = np.zeros([self.n_samples, M])
        starting_membership = 1. / M  # Begin with equal membership to each cluster
        for i, cso in enumerate(csos):
            memberships[cso, i] = 1
        for i, outlier in enumerate(outliers):
            memberships[outlier, M - 1] = 1
        for i, other in enumerate(others):
            memberships[other].fill(starting_membership)
        self.csos = csos
        self.outliers = outliers
        return memberships

    def _approximate_membership(self, memberships, weight_vector, knn_indices):
        iterations = 0
        error = 0
        while self.max_iterations is None or iterations < self.max_iterations:  # Iterative minimization of error function
            iterations += 1
            prev_memberships = memberships.copy()
            for l, knn in enumerate(knn_indices):
                if self.fixed.get(l):
                    continue
                memberships[l] = weight_vector.dot(prev_memberships[knn])

            error = np.square(memberships - prev_memberships).sum()
            if error < self.converge_threshold:
                self._print("** Convergence reached after {} iterations (error: {})".format(iterations, error))
                return memberships
        self._print("** Maximum of {} iterations reached (approximation error: {})".format(self.max_iterations, error))
        return memberships

    def _assign_single_membership(self, memberships):
        outliers = np.where(memberships[:, -1] == 1)[0]
        membership_indices = memberships[:, :-1].argmax(axis=1)
        membership_indices[outliers] = len(self.csos)
        return membership_indices

    def _assign_multiple_membership(self, memberships):
        multiple_memberships = []
        for i, x in enumerate(memberships[:, :-1]):
            multiple_memberships.append([(j, z) for j, z in enumerate(x) if z > self.min_member_threshold])
        return multiple_memberships

    def cluster(self, X):
        self.n_samples = len(X)
        self._print("** Starting FLAME clustering with {} samples".format(self.n_samples))
        knn_distances, knn_indices = self._find_knearest(X)
        densities = self._compute_densities(knn_distances)
        weight_vector = self._compute_weight_vector(knn_distances)
        objects = self._define_object_types(densities, knn_indices)
        self._print("** Object type distribution: \n- Cluster Supporting Objects: {}\n- Outliers: {}\n- Normal: {}\n"
              .format(len(objects[0]), len(objects[1]), len(objects[2])))
        initial_memberships = self._initialize_memberships(objects)
        self._print("** Number of clusters: {} + 1 (outliers)".format(self.num_clusters - 1))
        self.fuzzy_memberships = self._approximate_membership(initial_memberships, weight_vector, knn_indices)
        # self.multiple_memberships = self._assign_multiple_membership(self.fuzzy_memberships)
        self.single_memberships = self._assign_single_membership(self.fuzzy_memberships)
        self._print("Done")
