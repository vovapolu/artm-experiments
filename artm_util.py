# coding: utf-8

import os
import sys

HOME = os.getenv('BIGARTM_HOME', "/")
BIGARTM_PATH = os.path.join(HOME, 'bigartm')
BIGARTM_BUILD_PATH = os.path.join(BIGARTM_PATH, 'build')
sys.path.append(os.path.join(BIGARTM_PATH, 'python'))
os.environ['ARTM_SHARED_LIBRARY'] = os.path.join(BIGARTM_BUILD_PATH, 'src/artm/libartm.so')
