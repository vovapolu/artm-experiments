from collections import defaultdict
import artm_util
from artm import *
import numpy as np
import warnings
import pandas as pd
from scipy.spatial.distance import euclidean
from csvwriter import CsvWriter
import random
import db_manage

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


class Pool:
    def __init__(self, topics_filter, save_topics=False):
        self.save_topics = save_topics
        self.phi = None
        self.theta = None
        self.topics = []
        self.topics_words = None
        self.topics_filter = topics_filter
        self.marks = dict()

    def collect_topics(self, phi, theta):
        topics_count = len(self.topics)
        new_topics_names = ['topic{}'.format(idx) for idx in xrange(topics_count, topics_count + phi.shape[1])]
        phi.columns = new_topics_names
        theta.index = new_topics_names
        if topics_count == 0:
            self.topics_words = phi.index.values
            self.phi = phi
            self.theta = theta
        else:
            if (phi.index.values != self.topics_words).any():
                warnings.warn("Words in new topics differ from words in other topics.\nIgnoring these topics.")
            else:
                self.phi = pd.concat([self.phi, phi], axis=1)
                self.theta = pd.concat([self.theta, theta])

        new_topics = self.topics_filter.filter_topics(self.phi, self.topics + list(phi.columns))

        if not self.save_topics:
            self.phi = self.phi[new_topics]
            self.theta = self.theta.loc[new_topics]
            new_topics = ['topic{}'.format(idx)
                          for idx in xrange(len(new_topics))]
            self.phi.columns = new_topics
            self.theta.index = new_topics

        self.topics = new_topics

    def get_basic_phi(self):
        if self.save_topics:
            return self.phi[self.get_basic_topics()]
        else:
            return self.phi

    def get_basic_theta(self):
        if self.save_topics:
            return self.theta.loc[self.get_basic_topics()]
        else:
            return self.theta

    def get_basic_topics_count(self):
        return len(self.topics)

    def get_basic_topics(self):
        return self.topics

    def get_top_words_by_topic(self, topic, words_number=5):
        words = self.phi[topic].values
        return self.topics_words[np.argpartition(-words, words_number)[:words_number]]

    def get_next_topics(self, topics_number):
        return self.get_basic_topics()[:topics_number]

    def process_marks(self, marks):
        self.marks.update(marks)

    def get_marked_topics_count(self):
        return len(self.marks)

    #def get_marked_basic_topics_count(self):
    #    return len([topic for topic in self.get_basic_topics() if topic in self.marks])

    def save(self, filename, topics='all'):
        pass

class Experiment:

    _info_builder = [
        ('basic_topics_by_iteration', lambda exp, model: exp.topics_pool.get_basic_topics_count()),
    ]

    default_collection_passes = 20

    def __init__(self, models, topics_pool):
        """

        :param models: list of dicts,
            each dict contains information about model and must contain key 'model', which value is a BigARTM model
        :param topics_pool:
            instance of Pool class
        """
        self.all_models = models
        self.new_models_idxs = range(len(models))
        self.marks = dict()
        self.topics_pool = topics_pool
        self.info = defaultdict(list)

    def load_data(self, data_name):
        self.data_name = data_name
        self.batch_vectorizer = BatchVectorizer(data_path=data_name, data_format='batches')

    def add_models(self, new_models):
        self.new_models_idxs += range(len(self.all_models), len(self.all_models) + len(new_models))
        self.all_models += new_models

    def run(self):
        new_models = [self.all_models[model_idx] for model_idx in self.new_models_idxs]
        self.new_models_idxs = []
        for model in new_models:
            num_collection_passes = model['num_collection_passes'] if 'num_collection_passes' in model \
                else Experiment.default_collection_passes
            model_factor = model['factor'] if 'factor' in model else 1
            for current_num in xrange(model_factor):
                model['model'].fit_offline(batch_vectorizer=self.batch_vectorizer,
                                           num_collection_passes=num_collection_passes,
                                           num_document_passes=1)
                self.topics_pool.collect_topics(model['model'].get_phi(), model['model'].get_theta())

                for builder_option in Experiment._info_builder:
                    self.info[builder_option[0]].append(builder_option[1](self, model['model']))

        print "Total basic topics: {}".format(self.topics_pool.get_basic_topics_count())

    def get_info(self):
        return self.info

    def show_next_topics_batch(self, topic_batch_size):
        topics = self.topics_pool.get_next_topics(topic_batch_size)
        for topic in topics:
            print "{}:\n{}".format(topic, self.topics_pool.get_top_words_by_topic(topic))

    def show_all_themes(self):
        self.show_next_topics_batch(self.topics_pool.get_basic_topics_count())

    def save_dataset_to_navigator(self):
        '''
            Code was taken from here
            https://github.com/bigartm/bigartm-book/blob/master/BigartmNavigatorExample.ipynb
        '''

        id = 1

        with CsvWriter(open('modalities.csv', 'w')) as out:
            out << [dict(id=id, name='words')]

        with open('docword.{}.txt'.format(self.data_name)) as f:
            D = int(f.readline())
            W = int(f.readline())
            n = int(f.readline())
            ndw_s = [map(int, line.split()) for line in f.readlines()]
            ndw_s = [(d - 1, w - 1, cnt) for d, w, cnt in ndw_s]

        with CsvWriter(open('documents.csv', 'w')) as out:
            out << (
                dict(id=d,
                     title='Document #{}'.format(d),
                     slug='document-{}'.format(d),
                     file_name='.../{}'.format(d))
                for d in range(D)
            )

        with open('vocab.{}.txt'.format(self.data_name)) as f, CsvWriter(open('terms.csv', 'w')) as out:
            out << (
                dict(id=i,
                     modality_id=id,
                     text=line.strip())
                for i, line in enumerate(f)
            )

        with CsvWriter(open('document_terms.csv', 'w')) as out:
            out << (
                dict(document_id=d,
                     modality_id=id,
                     term_id=w,
                     count=cnt)
                for d, w, cnt in ndw_s
            )

    def save_next_topics_to_navigator(self):
        '''
            Code was taken from here
            https://github.com/bigartm/bigartm-book/blob/master/BigartmNavigatorExample.ipynb
        '''

        pwt = self.topics_pool.get_basic_phi().as_matrix()
        ptd = self.topics_pool.get_basic_theta().as_matrix()
        pd = 1.0 / ptd.shape[1]
        pt = (ptd * pd).sum(axis=1)
        pw = (pwt * pt).sum(axis=1)
        ptw = pwt * pt / pw[:, np.newaxis]
        pdt = ptd * pd / pt[:, np.newaxis]

        with CsvWriter(open('topics.csv', 'w')) as out:
            out << [dict(id=0,
                         level=0,
                         id_in_level=0,
                         is_background=False,
                         probability=1)]  # the single zero-level topic with id=0 is required
            out << (dict(id=1 + t,  # any unique ids
                         level=1,  # for a flat non-hierarchical model just leave 1 here
                         id_in_level=t,
                         is_background=False,  # if you have background topics, they should have True here
                         probability=p)
                    for t, p in enumerate(pt))

        with CsvWriter(open('topic_terms.csv', 'w')) as out:
            out << (dict(topic_id=1 + t,  # same ids as above
                         modality_id=1,
                         term_id=w,
                         prob_wt=pwt[w, t],
                         prob_tw=ptw[w, t])
                    for w, t in zip(*np.nonzero(pwt)))

        with CsvWriter(open('document_topics.csv', 'w')) as out:
            out << (dict(topic_id=1 + t,  # same ids as above
                         document_id=d,
                         prob_td=ptd[t, d],
                         prob_dt=pdt[t, d])
                    for t, d in zip(*np.nonzero(ptd)))

        with CsvWriter(open('topic_edges.csv', 'w')) as out:
            out << (dict(parent_id=0,
                         child_id=1 + t,
                         probability=p)
                    for t, p in enumerate(pt))



    def mark_topics(self, marks):
        self.marks.update(marks)

    def process_marks(self):
        self.topics_pool.process_marks(self.marks)
        self.marks = dict()

        print "Unmarked basic topics: {}".format(self.topics_pool.get_basic_topics_count() -
                                                 self.topics_pool.get_marked_basic_topics_count())

