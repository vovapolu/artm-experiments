import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from scipy.spatial import ConvexHull

class ConvexHullTopicsFilter:
    def __init__(self, eps=1e-3, iter_num=1, metric=euclidean, verbose=False):
        self.metric = metric
        self.eps = eps
        self.iter_num = iter_num
        self.verbose = verbose

    @staticmethod
    def project_points(points, dim=None):
        if dim is None:
            dim = 5
            #dim = min(max(int(np.log(len(points))), 2), 15)

        proj = GaussianRandomProjection(n_components=dim)
        return proj.fit_transform(points)

    def filter_topics(self, topics, used_topics):
        topics_idx = set()
        for i in xrange(self.iter_num):
            if self.verbose:
                print "Iteration {}".format(i)

            #if self.verbose:
                #print "Projecting points to {}-dimensional space...".format(dim)

            points = ConvexHullTopicsFilter.project_points(topics[used_topics].as_matrix().T)
            
            if self.verbose:
                print "Projecting to {}-dimensional space...".format(points.shape[1])
            
            if self.verbose:
                print "Building convex hull..."
            hull = ConvexHull(points, qhull_options='W{} C{}'.format(self.eps, self.eps))
            topics_idx.update(hull.vertices)

            filtered_topics_idx = set()
            for topic_idx1 in topics_idx:
                is_close = False
                for topic_idx2 in filtered_topics_idx:
                    if euclidean(points[topic_idx1], points[topic_idx2]) < self.eps:
                        is_close = True
                        break

                if not is_close:
                    filtered_topics_idx.add(topic_idx1)

            topics_idx = filtered_topics_idx

            if self.verbose:
                print "Chosen topics: {}".format(len(topics_idx))

        return [used_topics[topic_idx] for topic_idx in sorted(topics_idx)]
