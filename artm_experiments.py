from collections import defaultdict
import artm_util
import numpy as np
import warnings
import pandas as pd
from scipy.spatial import ConvexHull

class ConvexHullMerger:
    def __init__(self):
        self.convex_hull = None

    def fit(self, topics):
        self.convex_hull = ConvexHull(topics, incremental=True)

    def merge(self, topics):
        self.convex_hull.add_points(topics)

    def get_main_indices(self):
        return self.convex_hull.vertices

class Pool:
    def __init__(self, merger):
        self.topics_matrix = None
        self.merger = merger
        self.marks = dict()

    def collect_topics(self, phi):
        if self.topics_matrix is None:
            phi.columns = ['topic{}'.format(idx) for idx in xrange(phi.shape[1])]
            self.topics_matrix = phi
            self.merger.fit(self.topics_matrix.as_matrix().T)
        else:
            if list(phi.index) != list(self.topics_matrix.index):
                warnings.warn("Words in new topics differ from words in other topics.\nIgnoring these topics.")
            else:
                phi.columns = ['topic{}'.format(self.topics_matrix.shape[1] + idx) for idx in xrange(phi.shape[1])]
                self.topics_matrix = pd.concat([self.topics_matrix, phi], axis=1)
                self.merger.merge(phi.as_matrix().T)

    def get_topics_count(self):
        return self.topics_matrix.shape[1]

    def get_basic_topics_count(self):
        return len(self.merger.get_main_indices())

    def get_basic_topics(self):
        return np.array(list(self.topics_matrix))[self.merger.get_main_indices()]

    def get_derivative_topics_count(self):
        return self.get_topics_count() - self.get_basic_topics_count()

    def get_derivative_topics(self):
        derivative_idxs = np.delete(np.arange(self.get_topics_count()), self.merger.get_main_indices())
        return np.array(list(self.topics_matrix))[derivative_idxs]

    def get_top_words_by_topic(self, topic, words_number=20):
        topic_column = self.topics_matrix[topic].values()
        return self.topics_matrix.index.values[np.argpartition(topic_column, words_number)[:words_number]]

    def get_next_topics(self, topics_number):
        if self.topics_matrix is None:
            warnings.warn("Topics weren't initialized.")
            return None
        if topics_number > self.get_basic_topics_count():
            topics_number = self.get_basic_topics_count()

        topics = []
        for topic in self.get_basic_topics():
            if len(topics_number) >= topics_number:
                break
            if topic not in self.marks:
                topics.append(topic)

        return topics

    def process_marks(self, marks):
        self.marks.update(marks)

    def get_marked_topics_count(self):
        return len(self.marks)

    def get_marked_basic_topics_count(self):
        return len([topic for topic in self.get_basic_topics() if topic in self.marks])

    def save(self, filename, topics='all'):
        pass

class Experiment:

    _info_builder = [
        ('basic_topics_by_iteration', lambda exp, model: exp.topics_pool.get_basic_topics_count()),
        ('derivative_topics_by_iteration', lambda exp, model: exp.topics_pool.get_derivative_topics_count())
    ]

    default_collection_passes = 10

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

    def load_data(self, bathches_dir):
        self.batch_vectorizer = BatchVectorizer(data_path=bathches_dir, data_format='batches')

    def add_models(self, new_models):
        self.new_models_idxs += range(len(self.all_models), len(self.all_models) + len(new_models))
        self.all_models += new_models

    def run(self):
        new_models = [self.all_models[model_idx] for model_idx in self.new_models_idxs]
        self.new_models_idxs = []
        for model in new_models:
            num_collection_passes = model['num_collection_passes'] if 'num_collection_passes' in model \
                else Experiment.default_collection_passes
            model['model'].fit_offline(batch_vectorizer=self.batch_vectorizer,
                                       num_collection_passes=num_collection_passes,
                                       num_document_passes=1)
            self.topics_pool.collect_topics(model['model'].get_phi())

            for builder_option in Experiment._info_builder:
                self.info[builder_option[0]].append(builder_option[1](self, model['model']))

        print "Total basic topics: {}".format(self.topics_pool.get_basic_topics_count())

    def get_info(self):
        return self.info

    def show_next_topics_batch(self, topic_batch_size):
        for topic in self.topics_pool.get_next_topics(topic_batch_size):
            print topic, self.topics_pool.get_top_words_by_topic(topic)

    def save_next_topics_batch(self, filename):
        pass

    def mark_topics(self, marks):
        self.marks.update(marks)

    def process_marks(self):
        self.topics_pool.process_marks(self.marks)
        self.marks = dict()

        print "Unmarked basic topics: {}".format(self.topics_pool.get_basic_topics_count() -
                                                 self.topics_pool.get_marked_basic_topics_count())

