import artm_util
from artm import *

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
from csvwriter import CsvWriter
import csv
from subprocess import call, check_output
import re
from scipy.spatial.distance import euclidean

from greedy_methods import GreedyTopicsFilter, GreedyTopicsRanker
from convex_hull_methods import ConvexHullTopicsFilter
from optimize_methods import OptimizationTopicsFilter

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
        topics_count = 0 if self.phi is None else self.phi.shape[1]
        new_topics_names = ['topic{}'.format(idx)
                            for idx in xrange(topics_count, topics_count + phi.shape[1])]
        phi.columns = new_topics_names
        theta.index = new_topics_names
        if topics_count == 0:
            self.topics_words = phi.index.values
            self.phi = phi
            self.theta = theta
        else:
            if (phi.index.values != self.topics_words).any():
                warnings.warn("Words in new topics differ from words in other topics.\nIgnoring these topics.")
                return
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

    def get_all_topics(self):
        return self.phi.columns

    def get_top_words_by_topic(self, topic, words_number=5):
        words = self.phi[topic].values
        return self.topics_words[np.argpartition(-words, words_number)[:words_number]]

    def get_closest_basic_topic(self, topic, metric=euclidean):
        closest_topic = None
        closest_dist = None
        for basic_topic in self.get_basic_topics():
            dist = metric(self.phi[topic], self.phi[basic_topic])
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_topic = basic_topic

        return closest_topic

    def get_dist_between_topics(self, topic1, topic2, metric=euclidean):
        return metric(self.phi[topic1], self.phi[topic2])

    def get_next_topics(self, topics_number):
        topics = []
        for topic in self.get_basic_topics():
            if topic not in self.marks:
                topics.append(topic)
            if len(topics) >= topics_number:
                break

        return topics

    def process_marks(self, marks):
        self.marks.update(marks)

    def get_marked_topics_count(self):
        return len(self.marks)

class Experiment:

    _info_builder = [
        ('basic_topics_by_iteration', lambda exp, model: exp.topics_pool.get_basic_topics_count()),
    ]

    default_collection_passes = 20

    def __init__(self, topics_pool, models=list()):
        """

        :param models: list of dicts,
            each dict contains information about model and must contain key 'model', which value is a BigARTM model
        :param topics_pool:
            instance of Pool class
        """
        self.all_models = models
        self.new_models_idxs = range(len(models))
        self.assessments = dict()
        self.topics_pool = topics_pool
        self.info = defaultdict(list)

        # navigator
        self.dataset_id = None
        self.topic_model_id = None

    def load_data(self, data_name):
        self.data_name = data_name
        self.batch_vectorizer = BatchVectorizer(data_path=data_name, data_format='batches')

    def add_models(self, new_models):
        self.new_models_idxs += range(len(self.all_models), len(self.all_models) + len(new_models))
        self.all_models += new_models

    def collect_topics(self, phi, theta):
        self.topics_pool.collect_topics(phi, theta)

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
                self.collect_topics(model['model'].get_phi(), model['model'].get_theta())

                for builder_option in Experiment._info_builder:
                    self.info[builder_option[0]].append(builder_option[1](self, model['model']))
                # rewrite to cycle

        print("Total basic topics: {}".format(self.topics_pool.get_basic_topics_count()))

    def get_info(self):
        return self.info

    def show_topics(self, topics, show_top_words=True, show_closest_basic_topic=True,
                    sort_by_closest_topic=False):

        topics_info = []
        for topic in topics:
            s = [topic]
            if show_top_words:
                s.append('{}'.format(self.topics_pool.get_top_words_by_topic(topic)))
            if show_closest_basic_topic:
                s.append('{}'.format(self.topics_pool.get_closest_basic_topic(topic)))
            topics_info.append(s)

        if sort_by_closest_topic:
            topics_info = sorted(topics_info, key=lambda s: s[2])

        for info in topics_info:
            print ' | '.join(info)

    def show_all_topics(self, sort_by_closest_topic=False):
        self.show_topics(self.topics_pool.get_all_topics(), sort_by_closest_topic=sort_by_closest_topic)

    def show_basic_topics(self):
        self.show_topics(self.topics_pool.get_basic_topics(), show_closest_basic_topic=False)

    def show_next_topics_batch(self, topic_batch_size):
        topics = self.topics_pool.get_next_topics(topic_batch_size)
        for topic in topics:
            print("{}:\n{}".format(topic, self.topics_pool.get_top_words_by_topic(topic)))

    def show_all_themes(self):
        self.show_next_topics_batch(self.topics_pool.get_basic_topics_count())

    @staticmethod
    def get_navigator_home():
        return os.path.abspath(os.getenv('NAVIGATOR_HOME', '../tm_navigator'))

    @staticmethod
    def run_navigator(*args):
        cur_path = os.getcwd()
        os.chdir(Experiment.get_navigator_home())
        try:
            output = check_output('yes | ./db_manage.py ' + ' '.join(args), shell=True)
        finally:
            os.chdir(cur_path)
        return output

    def save_dataset_to_navigator(self):
        '''
            Code was taken from here
            https://github.com/bigartm/bigartm-book/blob/master/BigartmNavigatorExample.ipynb
        '''

        def in_dataset_folder(filename):
            return os.path.join(self.data_name, filename)

        id = 1

        with CsvWriter(open(in_dataset_folder('modalities.csv'), 'w')) as out:
            out << [dict(id=id, name='words')]

        with open(in_dataset_folder('docword.{}.txt'.format(self.data_name))) as f:
            D = int(f.readline())
            W = int(f.readline())
            n = int(f.readline())
            ndw_s = [map(int, line.split()) for line in f.readlines()]
            ndw_s = [(d - 1, w - 1, cnt) for d, w, cnt in ndw_s]

        with CsvWriter(open(in_dataset_folder('documents.csv'), 'w')) as out:
            out << (
                dict(id=d,
                     title='Document #{}'.format(d),
                     slug='document-{}'.format(d),
                     file_name='.../{}'.format(d))
                for d in range(D)
            )

        with open(in_dataset_folder('vocab.{}.txt'.format(self.data_name))) as f, \
                CsvWriter(open(in_dataset_folder('terms.csv'), 'w')) as out:
            out << (
                dict(id=i,
                     modality_id=id,
                     text=line.strip())
                for i, line in enumerate(f)
            )

        with CsvWriter(open(in_dataset_folder('document_terms.csv'), 'w')) as out:
            out << (
                dict(document_id=d,
                     modality_id=id,
                     term_id=w,
                     count=cnt)
                for d, w, cnt in ndw_s
            )

        print('Files saved.')

        output = Experiment.run_navigator('add_dataset')
        self.dataset_id = re.search('Added Dataset #(\d+)', output).group(1)
        Experiment.run_navigator('load_dataset', '--dataset-id', self.dataset_id,
                                 '--title', self.data_name, '-dir', os.path.abspath(self.data_name))

    def save_next_topics_batch_to_navigator(self, topic_batch_size):
        '''
            Code was taken from here
            https://github.com/bigartm/bigartm-book/blob/master/BigartmNavigatorExample.ipynb
        '''

        topics = self.topics_pool.get_next_topics(topic_batch_size)
        topics_ids = [int(topic[5:]) for topic in topics]  # topic123 -> 123

        pwt = self.topics_pool.get_basic_phi()[topics].as_matrix()
        ptd = self.topics_pool.get_basic_theta().loc[topics].as_matrix()
        pd = 1.0 / ptd.shape[1]
        pt = (ptd * pd).sum(axis=1)
        pw = (pwt * pt).sum(axis=1)
        ptw = pwt * pt / pw[:, np.newaxis]
        pdt = ptd * pd / pt[:, np.newaxis]

        def in_dataset_folder(filename):
            return os.path.join(self.data_name, filename)

        with CsvWriter(open(in_dataset_folder('topics.csv'), 'w')) as out:
            out << [dict(id=0,
                         level=0,
                         id_in_level=0,
                         is_background=False,
                         probability=1)]  # the single zero-level topic with id=0 is required
            out << (dict(id=1 + topics_ids[t],  # any unique ids
                         level=1,  # for a flat non-hierarchical model just leave 1 here
                         id_in_level=topics_ids[t],
                         is_background=False,  # if you have background topics, they should have True here
                         probability=p)
                    for t, p in enumerate(pt))

        with CsvWriter(open(in_dataset_folder('topic_terms.csv'), 'w')) as out:
            out << (dict(topic_id=1 + topics_ids[t],  # same ids as above
                         modality_id=1,
                         term_id=w,
                         prob_wt=pwt[w, t],
                         prob_tw=ptw[w, t])
                    for w, t in zip(*np.nonzero(pwt)))

        with CsvWriter(open(in_dataset_folder('document_topics.csv'), 'w')) as out:
            out << (dict(topic_id=1 + topics_ids[t],  # same ids as above
                         document_id=d,
                         prob_td=ptd[t, d],
                         prob_dt=pdt[t, d])
                    for t, d in zip(*np.nonzero(ptd)))

        with CsvWriter(open(in_dataset_folder('topic_edges.csv'), 'w')) as out:
            out << (dict(parent_id=0,
                         child_id=1 + topics_ids[t],
                         probability=p)
                    for t, p in enumerate(pt))

        if self.dataset_id is None:
            warnings.warn("Dataset wasn't loaded to navigator.")
        else:
            output = Experiment.run_navigator('add_topicmodel', '--dataset-id', self.dataset_id)
            self.topic_model_id = re.search('Added Topic Model #(\d+) for Dataset #(\d+)', output).group(1)
            Experiment.run_navigator('load_topicmodel', '--topicmodel-id', self.topic_model_id,
                                     '--title', self.data_name, '-dir', os.path.abspath(self.data_name))

    def load_assessments_from_navigator(self):

        def in_dataset_folder(filename):
            return os.path.join(self.data_name, filename)

        Experiment.run_navigator('dump_assessments', '-dir', os.path.abspath(self.data_name),
                                 '-m', self.topic_model_id)
        with open(in_dataset_folder('topic_assessments.csv')) as assessments:
            reader = csv.DictReader(assessments)
            for row in reader:
                topic = 'topic' + row['topic_id']
                assessment = row['value']
                if assessment != '':
                    self.assessments[topic] = int(assessment)

    def assess_topics(self, assessments):
        self.assessments.update(assessments)

    def show_assessments(self):
        for topic, assessment in self.assessments.iteritems():
            print("{}: {}".format(topic, assessment))

    def process_assessments(self):
        self.topics_pool.process_marks(self.assessments)
        self.assessments = dict()

        print("Unmarked basic topics: {}".format(self.topics_pool.get_basic_topics_count() -
                                                 self.topics_pool.get_marked_basic_topics_count()))

