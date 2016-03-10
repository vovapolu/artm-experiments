import random
import numpy as np
from scipy.spatial.distance import euclidean

def greedy_most_distans_points_search(points, used_points, stop_condition, metric=euclidean):
    taken_points = []
    first_point = random.choice(used_points)
    taken_points.append(first_point)

    def find_most_distant_point():
        max_dist = 0
        max_point = None
        for point in used_points:
            if point not in taken_points:
                min_used_dist = None
                for taken_point in taken_points:
                    dist = metric(points[point].values, points[taken_point].values)
                    if min_used_dist is None or dist < min_used_dist:
                        min_used_dist = dist

                if min_used_dist > max_dist:
                    max_dist = min_used_dist
                    max_point = point

        return max_point, max_dist

    taken_points.append(find_most_distant_point()[0])
    taken_points.remove(first_point)

    dists = []
    while len(taken_points) < len(used_points):
        point, dist = find_most_distant_point()
        if stop_condition(dist, dists):
            break
        taken_points.append(point)
        dists.append(dist)

    return taken_points

class GreedyTopicsRanker:
    def __init__(self):
        self.topics = None

    def fit(self, topics):
        self.topics = topics

    def rank_next_topics(self, topics_cnt):
        def cnt_condition(dist, last_dists):
            return len(last_dists) >= topics_cnt

        topics_idxs = greedy_most_distans_points_search(self.topics.as_matrix().T, stop_condition=cnt_condition)
        return self.topics.ix[:, topics_idxs]


class GreedyTopicsFilter:
    def __init__(self, sigma=4, mean_cnt=5):
        self.sigma = sigma
        self.mean_cnt = mean_cnt

    def filter_topics(self, topics, used_topics):
        def dist_condition(dist, last_dists):
            recent_dists = last_dists[-self.mean_cnt:]
            return len(last_dists) >= self.mean_cnt and \
                (np.abs(np.mean(recent_dists) - dist)) / np.std(recent_dists) > self.sigma

        return greedy_most_distans_points_search(topics, used_topics, stop_condition=dist_condition)
