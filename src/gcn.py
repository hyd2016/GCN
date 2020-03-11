# coding=utf-8
import numpy as np
import networkx as nx
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from scipy import stats
import sys


class Graph:
    def __init__(self, nx_G, list_G, test_G, is_directed):
        self.G = nx_G
        self.is_directed = is_directed
        self.list_G = list_G
        self.test_G = test_G
        self.n = len(nx_G)
        self.edge_info = []

    def get_time_matrix(self, lam, time_count):
        # 初始化权重，lam是每个时刻图对下一个时刻的影响参数，指数衰减
        matrix_time = np.zeros((self.n, self.n))
        for g in self.list_G:
            for edge in g.edges():
                if g[edge[0]][edge[1]]['weight'] > 1:
                    g[edge[0]][edge[1]]['weight'] = 1
        for i, g in enumerate(self.list_G):
            t = np.array(nx.adjacency_matrix(g).todense())
            matrix_time = matrix_time + lam ** (time_count - i) * t
        return matrix_time

    def get_similarity(self, lam, time_count):
        Matrix_time = self.get_time_matrix(lam, time_count)
        Matrix_similarity = Matrix_time * Matrix_time
        max_num = np.max(Matrix_similarity)
        return Matrix_similarity / max_num

    def predict(self, lam, time_count):
        matrix_p = self.get_time_matrix(lam, time_count)

        fpr, tpr, thresholds = metrics.roc_curve(np.array(nx.adjacency_matrix(self.test_G).todense()).flatten(),
                                                 matrix_p.flatten(), pos_label=1)
        plt.plot(fpr, tpr, marker='o')
        plt.show()
        auc_score = roc_auc_score(np.array(nx.adjacency_matrix(self.test_G).todense()).flatten(), matrix_p.flatten())
        print auc_score
        return
