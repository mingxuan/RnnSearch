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

from trans_model import Translate
# Get the arguments

if __name__ == "__main__":
    config = getattr(configurations, 'get_config_cs2en')()

    source = T.lmatrix('source')
    target = T.lmatrix('target')
    source_mask = T.matrix('source_mask')
    target_mask = T.matrix('target_mask')

    trans = Translate(**config)
    trans.apply(source.T, source_mask.T, target.T, target_mask.T)
    cost = trans.cost

    #trans.load(config['saveto']+'/params.npz')
    params = trans.params

    for value in params:
        logger.info('    {:15}: {}'.format(value.get_value().shape, value.name))

    grade = T.grad(cost, params)

    # add step clipping
    if config['step_clipping'] > 0.:
        grad = step_clipping(params, grade, config['step_clipping'])

    updates = adadelta(params, grade)

    logger.info('begin to build translation model : tr_fn')
    tr_fn = theano.function([source, source_mask, target, target_mask],
                            [cost], updates=updates)
    logger.info('end build translation model : tr_fn')

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

    tr_stream = get_tr_stream(**config)
    dev_stream = get_dev_stream(**config)
    logger.info('start training!!!')
    batch_count = 0

    val_time = 0
    best_score = 0.
    for epoch in range(config['max_epoch']):
        for tr_data in tr_stream.get_epoch_iterator():
            batch_count += 1
            tr_fn(*tr_data)

            # sample
            if batch_count % config['sampling_freq'] == 0:
                trans_sample(tr_data[0], tr_data[2], f_init, f_next, config['hook_samples'],
                             src_vocab_reverse, trg_vocab_reverse, batch_count)

            # trans valid data set
            if batch_count > config['val_burn_in'] and batch_count % config['bleu_val_freq'] == 0:
                logger.info('[{}]: {} has been tackled and start translate val set!'.format(epoch, batch_count))
                val_time += 1
                val_save_out = '{}.{}'.format(config['val_set_out'], val_time)
                val_save_file = open(val_save_out, 'w')
                data_iter = dev_stream.get_epoch_iterator()
                trans_res = multi_process_sample(data_iter, f_init, f_next, k=12, vocab=trg_vocab_reverse, process=1)
                val_save_file.writelines(trans_res)
                val_save_file.close()
                logger.info('[{}]: {} times val has been translated!'.format(epoch, val_time))

                bleu_score = valid_bleu(config['eval_dir'], val_save_out)
                os.rename(val_save_out, "{}.{}.txt".format(val_save_out, bleu_score))
                if bleu_score > best_score:
                    trans.savez(config['saveto']+'/params.npz')
                    best_score = bleu_score
                    logger.info('epoch:{}, batchs:{}, bleu_score:{}'.format(
                        epoch, batch_count, best_score))


