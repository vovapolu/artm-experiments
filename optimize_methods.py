import numpy as np
import random
import scipy.optimize
from scipy.spatial.distance import sqeuclidean


def euclidean_metric(w, X, y):
    return sqeuclidean(X.dot(w), y)

def jac_euclidean_metric(w, X, y):
    return 2 * (X.T.dot(X.dot(w) - y))

def cosine_metric(w, X, y):
    pass

def jac_cosine_metric(w, X, y):
    pass

class OptimizationTopicsFilter:

    def __init__(self, eps=1e-3, metric=euclidean_metric, jac_metric=jac_euclidean_metric):
        self.eps = eps
        self.metric = metric
        self.jac_metric = jac_metric

    def filter_topics(self, topics, used_topics):
        shuffled_topics = list(used_topics)
        random.shuffle(shuffled_topics)

        res_topics = list(used_topics)

        for topic in shuffled_topics:
            y = topics[topic].as_matrix().ravel()
            res_topics.remove(topic)
            X = topics[res_topics].as_matrix()
            func = lambda w: self.metric(w, X, y)
            jac = lambda w: self.jac_metric(w, X, y)

            w_init = np.ones(X.shape[1]) / X.shape[1]

            res = scipy.optimize.minimize(func, w_init, method='SLSQP', jac=jac,
                                          bounds=[(0.0, 1.0)] * X.shape[1],
                                          constraints={'type': 'eq',
                                                       'fun': lambda w: np.sum(w) - 1,
                                                       'jac': lambda w: np.ones(X.shape[1])},
                                          options={'maxiter': 100})

            print res.fun
            print X.shape[1]

            if res.fun > self.eps:
                res_topics.append(topic)

        return res_topics
