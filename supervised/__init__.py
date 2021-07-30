# -*- coding: utf-8 -*-
# Python Keyphrase Extraction toolkit: unsupervised models

from __future__ import absolute_import

from supervised.api import SupervisedLoadFile
from supervised.feature_based.kea import Kea
from supervised.feature_based.topiccorank import TopicCoRank
from supervised.feature_based.wingnus import WINGNUS
from supervised.neural_based.seq2seq import Seq2Seq