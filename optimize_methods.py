import numpy as np
import random
import scipy.optimize
from scipy.spatial.distance import sqeuclidean, euclidean
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from numpy.linalg import norm


def euclidean_metric(w, X, y):
    return sqeuclidean(X.dot(w), y)

def jac_euclidean_metric(w, X, y):
    return 2 * (X.T.dot(X.dot(w) - y))

def cosine_metric(w, X, y):
    return cosine_distances([X.dot(w)], [y])[0, 0]

def jac_cosine_metric(w, X, y):
    v = X.dot(w)
    u = X.T.dot(y)
    normy = norm(y)
    normv = norm(v)
    return -u / normv / normy + v.dot(y) * X.T.dot(v) / (normv ** 1.5) / normy

class OptimizationTopicsFilter:

    def __init__(self, eps=1e-3, metric=euclidean_metric, jac_metric=jac_euclidean_metric, verbose=False):
        self.eps = eps
        self.metric = metric
        self.jac_metric = jac_metric
        self.verbose = verbose
        self.projections = dict()

    def get_dist(self, topic_name, topic, topics, used_topics):
        X = topics[used_topics].as_matrix()
        func = lambda w: self.metric(w, X, topic)
        jac = lambda w: self.jac_metric(w, X, topic)

        w_init = np.ones(X.shape[1]) / X.shape[1]

        res = scipy.optimize.minimize(func, w_init, method='SLSQP', jac=jac,
                                      bounds=[(0.0, 1.0)] * X.shape[1],
                                      constraints={'type': 'eq',
                                                   'fun': lambda w: np.sum(w) - 1,
                                                   'jac': lambda w: np.ones(X.shape[1])},
                                      options={'maxiter': 5},
                                      tol=1e-4)

        self.projections[topic_name] = X.dot(res.x)

        return res

    def filter_topics(self, topics, used_topics):
        shuffled_topics = list(used_topics)
        random.shuffle(shuffled_topics)

        res_topics = list(used_topics)
        self.res_vals = dict()

        for topic in shuffled_topics:
            y = topics[topic].as_matrix().ravel()
            res_topics.remove(topic)
            res = self.get_dist(topic, y, topics, res_topics)

            self.res_vals[topic] = res.fun

            if res.fun > self.eps:
                res_topics.append(topic)

        if self.verbose:
            self.plot_hist()

        return res_topics #, res_vals, cnt_vals, shuffled_topics

    def plot_hist(self, topics='all'):
        if topics == 'all':
            topics = self.res_vals.keys()
        vals = [self.res_vals[topic] for topic in topics]
        n, bins, _ = plt.hist(np.log10(vals), bins=20)
        plt.xlabel('Log10 dist')
        plt.ylabel('Number of topics')
        plt.show()
        return n, bins
