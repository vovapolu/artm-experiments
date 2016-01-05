from collections import defaultdict
import artm_util
import numpy as np
import warnings

class Pool:
    def __init__(self):
        self.topics_matrix = None

    def collect_topics(self, phi):
        if self.topics_matrix is None:
            self.topics_matrix = phi
        else:
            if phi.shape[0] != self.topics_matrix.shape[0]:
                warnings.warn("Number of words in new topics differs from other topics.\nIgnoring these topics.")
            else:
                self.topics_matrix = np.hstack((self.topics_matrix, phi))

    def get_basic_topics_count(self):
        pass

    def get_basic_topics(self):
        pass

    def get_derivative_topics_count(self):
        pass

    def get_derivative_topics(self):
        pass

    def get_most_valuable_topics(self, topics_number):
        if self.topics_matrix is None:
            warnings.warn("Topics weren't initialized.")
            return None
        if topics_number > self.topics_matrix.shape[1]:
            topics_number = self.topics_matrix.shape[1]
        return np.arange(topics_number)

    def process_marks(self, marks):
        pass

    def save(self, filename):
        pass

class Experiment:

    _info_builder = [
        ('basic_topics_by_iteration', lambda exp, model: exp.topics_pool.get_basic_topics_count()),
        ('derivative_topics_by_iteration', lambda exp, model: exp.topics_pool.get_derivative_topics_count())
    ]

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

    def run(self, topics_batch_size):
        self.topic_batch_size = topics_batch_size
        new_models = [self.all_models[model_idx] for model_idx in self.new_models_idxs]
        self.new_models_idxs = []
        for model in new_models:
            num_collection_passes = model['num_collection_passes'] if 'num_collection_passes' in model else 10
            model['model'].fit_offline(batch_vectorizer=self.batch_vectorizer,
                                       num_collection_passes=num_collection_passes,
                                       num_document_passes=1)
            self.topics_pool.collect_topics(model['model'].phi_.as_matrix())
            for builder_option in Experiment._info_builder:
                self.info[builder_option[0]].append(builder_option[1](self, model['model']))

    def get_info(self):
        return self.info

    def show_topics_batch(self):
        for topic in self.topics_pool.get_most_valuable_topics(self.topic_batch_size):
            somehow_print_topic(topic)

    def save_topics_batch(self, filename):
        pass

    def mark_topics(self, marks):
        self.marks.update(marks)

    def process_marks(self):
        self.topics_pool.process_marks(self.marks)

