import numpy as np
import copy
import random
import clustering_methods
import logging
from sklearn import preprocessing

class Method:
    """
    """
    def __init__(self,dataset,algorithm,psi,t,h):

        self.dataset = dataset
        self.algorithm = algorithm
        self.IK_type = 'inne'
        self.psi = psi
        self.t = t
        self.h = h


    def IK_fm_dot(self,X,psi,t,):
        """
        """

        onepoint_matrix = np.zeros((X.shape[0], (int)(t * psi)), dtype=int)
        x_index=np.arange(len(X))
        for time in range(t):
            sample_num = self.psi  #
            sample_list = [p for p in range(len(X))]  # [0, 1, 2, 3]
            sample_list = random.sample(sample_list, sample_num)  # [1, 2]
            sample = X[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
            # sim
            point2sample =np.dot(X,sample.T)
            min_dist_point2sample = np.argmax(point2sample, axis=1)+time*psi
        # dis
        #  from sklearn.metrics.pairwise import euclidean_distances
        #  point2sample =euclidean_distances(X,sample)
        #  min_dist_point2sample = np.argmin(point2sample, axis=1)+time*psi


            onepoint_matrix[x_index,min_dist_point2sample]=1

        return onepoint_matrix
    def IK_inne_fm(self,X, psi, t=100):
        onepoint_matrix = np.zeros((X.shape[0], (int)(t * psi)), dtype=int)
        for time in range(t):
            sample_num = self.psi  #
            sample_list = [p for p in range(len(X))]
            sample_list = random.sample(sample_list, sample_num)
            sample = X[sample_list, :]

            tem1 = np.dot(np.square(X), np.ones(sample.T.shape))  # n*psi
            tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
            point2sample = tem1 + tem2 - 2 * np.dot(X, sample.T)  # n*psi

            # tem = np.dot(np.square(sample), np.ones(sample.T.shape))
            # sample2sample = tem + tem.T - 2 * np.dot(sample, sample.T)
            sample2sample = point2sample[sample_list, :]
            row, col = np.diag_indices_from(sample2sample)
            sample2sample[row, col] = 1e9
            radius_list = np.min(sample2sample, axis=1)  # 每行的最小值形成一个行向量

            min_point2sample_index = np.argmin(point2sample, axis=1)
            min_dist_point2sample = min_point2sample_index + time * psi
            point2sample_value = point2sample[range(len(onepoint_matrix)), min_point2sample_index]
            ind = point2sample_value < radius_list[min_point2sample_index]
            onepoint_matrix[ind, min_dist_point2sample[ind]] = 1
        return onepoint_matrix

    def create_adj_avg(self,adj_mat):
        '''
        create adjacency
        '''
        np.fill_diagonal(adj_mat, 0)

        adj = copy.deepcopy(adj_mat)
        deg = np.sum(adj, axis=1)
        deg[deg == 0] = 1
        deg = (1/ deg) * 0.5
        deg_mat = np.diag(deg)
        adj = deg_mat.dot(adj_mat)
        np.fill_diagonal(adj, 0.5)
        return adj


    def WL_noconcate_fast(self, features, normalized_adj_mat):

        embedding = np.dot(normalized_adj_mat, features)

        return embedding


    def get_embedding(self,adj_mat,node_features,num_of_clusters, true_labels):
        normalized_adj_mat0 = self.create_adj_avg(adj_mat)
        if self.algorithm == 'WL':
            embedding = node_features.copy()
            for i in range(self.h):
                embedding = self.WL_noconcate_fast(embedding, normalized_adj_mat)
                self.do_clustering(embedding, num_of_clusters, true_labels)
        if self.algorithm == 'WDK':
            embedding = self.IK_fm_dot(node_features, self.psi, self.t)
            for i in range(self.h):
                embedding = self.WL_noconcate_fast(embedding, normalized_adj_mat)

        if self.algorithm == 'mWDK':
            embedding = node_features.copy()
            embedding1 = copy.deepcopy(node_features)

            for i in range(self.h):

                embedding = self.IK_fm_dot(embedding, self.psi, self.t)
                from sklearn.metrics.pairwise import linear_kernel

                embedding = self.WL_noconcate_fast(embedding, normalized_adj_mat0)
                new = linear_kernel(embedding)
                tmp = adj_mat * new
                #
                normalized_adj_mat = self.create_adj_avg(tmp)

                embedding1 = self.WL_noconcate_fast(embedding1, normalized_adj_mat)
                embedding1 = preprocessing.normalize(embedding1, norm='l2',axis=0)
                self.do_clustering(embedding, num_of_clusters, true_labels)
        return embedding



    def do_clustering(self,embedding, num_of_clusters, true_labels):
        clustering_methods.sc_linear(self.dataset,embedding, num_of_clusters, true_labels)