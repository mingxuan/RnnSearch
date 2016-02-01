import theano
import theano.tensor as T
import os

from utils import adadelta, step_clipping
from stream import get_tr_stream, get_dev_stream, ensure_special_tokens
import logging
import configurations
from sample import trans_sample, multi_process_sample, valid_bleu
import cPickle as pickle
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import numpy

from trans_model import Translate
# Get the arguments

if __name__ == "__main__":
    config = getattr(configurations, 'get_config_cs2en')()
    tr_stream = get_tr_stream(**config)
    logger.info('start training!!!')
    batch_count = 0

    val_time = 0
    best_score = 0.
    for epoch in range(3):
        print epoch
        for tr_data in tr_stream.get_epoch_iterator():
            batch_count += tr_data[0].shape[0]
        print batch_count


