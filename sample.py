import numpy
import copy
from functools import partial
from multiprocessing import Pool
import os
import subprocess
import re

def _index2sentence(vec, dic):
    r = [dic[index] for index in vec]
    return " ".join(r)

def gen_sample(x, f_init, f_next,  k=10, maxlen=40, vocab=None, normalize=True):

    # k is the beam size we have
    x = numpy.asarray(x, dtype='int64')
    if x.ndim == 1:
        x = x[None, :]
    x = x.T
    eos_id = len(vocab) - 1
    bos_id = 0

    sample = []
    sample_score = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = [-1]  # indicator for the first target word (bos target)

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_state = ret[0], ret[1]

        cand_scores = hyp_scores[:, None] - numpy.log(next_p)
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]

        voc_size = next_p.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
        new_hyp_states = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []

        for idx in xrange(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == eos_id:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
        hyp_scores = numpy.array(hyp_scores)
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = numpy.array([w[-1] for w in hyp_samples])
        next_state = numpy.array(hyp_states)

    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    if normalize:
        lengths = numpy.array([len(s) for s in sample])
        sample_score = sample_score/lengths
    sidx = numpy.argmin(sample_score)

    best_trans = sample[sidx]
    best_trans = filter(lambda item:item!=eos_id and item!=bos_id, best_trans)
    if vocab is not None:
        best_trans = _index2sentence(best_trans, vocab)
    return best_trans


def trans_sample(s, t, f_init, f_next,  hook_samples, src_vocab_reverse, trg_vocab_reverse, batch_count):
    hook_samples = min(hook_samples,s.shape[0])
    eos_id_src = len(src_vocab_reverse) - 1
    eos_id_trg = len(trg_vocab_reverse) - 1
    for index in range(hook_samples):
        s_filter = filter(lambda x:x!=eos_id_src, s[index])+[eos_id_src]
        trans = gen_sample(s_filter, f_init, f_next,  k=2, vocab=trg_vocab_reverse)
        print "translation sample {}".format(batch_count)
        print "[src] %s" % _index2sentence(s_filter, src_vocab_reverse)
        print "[ref] %s" % _index2sentence(filter(lambda x:x!=eos_id_trg, t[index])+[eos_id_trg], trg_vocab_reverse)
        print "[trans] %s" % trans
        print


def multi_process_sample(x_iter, f_init, f_next,  k=10, maxlen=50, vocab=None, normalize=True, process=5):
    partial_func = partial(gen_sample, f_init=f_init, f_next=f_next,
                           k=k, maxlen=maxlen, vocab=vocab, normalize=normalize)
    if process>1:
        pool = Pool(process)
        trans_res = pool.map(partial_func, x_iter)
    else:
        trans_res=map(partial_func, x_iter)

    trans_res = ['{}\n'.format(item) for item in trans_res]
    return trans_res

def valid_bleu(eval_dir, valid_out):
    child = subprocess.Popen('sh run.sh ../{}'.format(valid_out),
                             cwd=eval_dir,
                             shell=True, stdout=subprocess.PIPE)
    bleu_out = child.communicate()
    child.wait()
    bleu_pattern = re.search(r'BLEU score = (0\.\d+)', bleu_out[0])
    bleu_score = float(bleu_pattern.group(1))
    return bleu_score

if __name__=="__main__":
    import sys
    res = valid_bleu(sys.argv[1], sys.argv[2])
    print res


