import theano.tensor as T

from stream import get_tr_stream, get_dev_stream, ensure_special_tokens
import logging
import configurations
from sample import  multi_process_sample, gen_sample
import cPickle as pickle
logger = logging.getLogger(__name__)

from trans_model import Translate
# Get the arguments
import sys
if __name__ == "__main__":
    config = getattr(configurations, 'get_config_cs2en')()

    trans = Translate(**config)
    params = trans.params
    print params[0].get_value().sum()


    logger.info('begin to build sample model : f_init, f_next')
    f_init, f_next = trans.build_sample()
    logger.info('end build sample model : f_init, f_next')

    src_vocab = pickle.load(open(config['src_vocab']))
    trg_vocab = pickle.load(open(config['trg_vocab']))
    src_vocab = ensure_special_tokens(src_vocab,
                                      bos_idx=0, eos_idx=config['src_vocab_size'] - 1,
                                      unk_idx=config['unk_id'])
    trg_vocab = ensure_special_tokens(trg_vocab,
                                      bos_idx=0, eos_idx=config['src_vocab_size'] - 1,
                                      unk_idx=config['unk_id'])
    trg_vocab_reverse = {index: word for word, index in trg_vocab.iteritems()}
    src_vocab_reverse = {index: word for word, index in src_vocab.iteritems()}
    logger.info('load dict finished ! src dic size : {} trg dic size : {}.'.format(len(src_vocab), len(trg_vocab)))

    val_set=sys.argv[1]
    val_save_out = sys.argv[2]
    config['val_set']=val_set
    dev_stream = get_dev_stream(**config)
    logger.info('start training!!!')
    trans.load(config['saveto']+'/params.npz')

    print params[0].get_value().sum()

    data_iter = dev_stream.get_epoch_iterator()
    for data in data_iter:
        trans = gen_sample(data, f_init, f_next,  k=3, vocab=trg_vocab_reverse)
        print data
        print trans
    #trans = multi_process_sample(data_iter, f_init, f_next, k=3, vocab=trg_vocab_reverse, process=1)
    #val_save_file.writelines(trans)
    #val_save_file.close()





