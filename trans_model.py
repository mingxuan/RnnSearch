import numpy
import theano
import theano.tensor as T
from collections import OrderedDict
from model import GRU, Lookup_table, BiGRU, LogisticRegression
from utils import param_init


def _p(pp, name):
    return '%s_%s' % (pp, name)


class Attention(object):

    def __init__(self, s_in, t_in, prefix='Attention', **kwargs):
        self.params = []
        self.s_in = s_in
        self.t_in = t_in
        self.align_size = t_in
        self.prefix = prefix

        self.Wa = param_init().param((self.t_in, self.align_size), name=_p(prefix, 'Wa'))
        #self.v = param_init().param((self.align_size,), init_type='constant',
                                    #name=_p(prefix, 'v'), scale=0.001)

        self.v = param_init().param((self.align_size,), name=_p(prefix, 'v'))
        self.params += [self.Wa, self.v]

    def apply(self, source, source_mask, source_x, attention):
        if source.ndim != 3 or attention.ndim != 2:
            raise NotImplementedError

        align_matrix = T.tanh(source_x + T.dot(attention, self.Wa)[None, :, :])
        align = theano.dot(align_matrix, self.v)
        align = T.exp(align - align.max(axis=0, keepdims=True))
        if source_mask:
            align = align * source_mask
            normalization = align.sum(axis=0) + T.all(1 - source_mask, axis=0)
        else:
            normalization = align.sum(axis=0)
        align = align/normalization
        self.output = (T.shape_padright(align) * source).sum(axis=0)

        return self.output

class Decoder(GRU):

    def __init__(self, n_in, n_hids, c_hids, prefix='Decoder', **kwargs):
        kwargs['c_hids'] = c_hids
        kwargs['max_out'] = True
        kwargs['merge'] = True
        super(Decoder, self).__init__(n_in, n_hids, with_contex=True, prefix=prefix, **kwargs)
        self.attention_layer = Attention(self.c_hids, self.n_hids)
        self.params.extend(self.attention_layer.params)
        self._init_params2()

    def _init_params2(self):

        f = lambda name: _p(self.prefix, name)
        n_hids = self.n_hids
        size_hh = (n_hids, n_hids)

        self.W_hz2 = param_init().param(size_hh, 'orth', name=f('W_hz2'))
        self.W_hr2 = param_init().param(size_hh, 'orth', name=f('W_hr2'))
        self.W_hh2 = param_init().param(size_hh, 'orth', name=f('W_hh2'))
        self.b_z2 = param_init().param((n_hids,), name=f('b_z2'))
        self.b_r2 = param_init().param((n_hids,), name=f('b_r2'))
        self.b_h2 = param_init().param((n_hids,), name=f('b_h2'))

        self.Ws = param_init().param((self.c_hids, self.n_hids), name=f('Ws'))
        self.bs = param_init().param((self.n_hids,), name=f('bs'))
        self.params += [self.W_hz2, self.W_hr2, self.W_hh2,
                        self.b_z2, self.b_r2, self.b_h2, self.Ws, self.bs]


    def init_state(self, ctx, x_mask=None):
        if x_mask:
            ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]
        else:
            ctx_mean = ctx.mean(0)

        state = T.tanh(T.dot(ctx_mean, self.W_c_init) + self.b_init)

        return state

    def _forward(self, state_below, mask_below, context, c_mask):

        if state_below.ndim == 3 and context.ndim == 3:
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError

        init_state = self.init_state(context)
        context_x = T.dot(context, self.Ws) + self.bs
        non_sequences = [context, c_mask, context_x]
        rval, updates = theano.scan(self._step_forward_with_attention,
                                    sequences=[state_below, mask_below],
                                    outputs_info=[init_state, None],
                                    non_sequences=non_sequences,
                                    n_steps=n_steps
                                    )
        self.output = rval[0]
        self.attended = rval[1]
        return self.output, self.attended

    def _step_forward_with_attention(self, x_t, x_m, h_tm1, c, c_mask, c_x):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        '''
        # attended = self.attention_layer.apply(c, c_mask, h_tm1)
        # c_z = theano.dot(attended, self.W_cz)
        # c_r = theano.dot(attended, self.W_cr)
        # c_h = theano.dot(attended, self.W_ch)

        # return [self._step_forward_with_context(x_t, x_m, h_tm1, c_z, c_r, c_h), attended]

        #### new arc
        h1 = self._step_forward(x_t, x_m, h_tm1)
        attended = self.attention_layer.apply(c, c_mask, c_x,  h1 )
        z = T.nnet.sigmoid(theano.dot(attended, self.W_cz)
                           + theano.dot(h1, self.W_hz2) + self.b_z2)
        r = T.nnet.sigmoid(theano.dot(attended, self.W_cr)
                           + theano.dot(h1, self.W_hr2) + self.b_r2)
        c_h = theano.dot(attended, self.W_ch)
        h2 = T.tanh((T.dot(h1, self.W_hh2) + self.b_h2) * r + c_h)
        h2 = h1 * z + (1. - z) * h2
        if x_m:
            h2 = x_m[:, None] * h2 + (1. - x_m)[:, None] * h1
        return h2, attended

    def apply(self, state_below, mask_below, context, c_mask):
        hiddens, attended = self._forward(state_below, mask_below, context, c_mask)

        combine = T.concatenate([state_below, hiddens, attended], axis=2)

        if self.max_out:
            merge_out = theano.dot(combine, self.W_m) + self.b_m
            merge_out = merge_out.reshape((merge_out.shape[0],
                                           merge_out.shape[1],
                                           merge_out.shape[2]/2,
                                           2), ndim=4).max(axis=3)

        else:
            merge_out = T.tanh(theano.dot(combine, self.W_m) + self.b_m)

        return merge_out * mask_below[:, :, None]


    def next_state_merge(self, y_emb, cur_state, c, c_x):
        next_state, attended = self._step_forward_with_attention(x_t=y_emb,
                                                                 x_m=None,
                                                                 h_tm1=cur_state,
                                                                 c=c,
                                                                 c_mask=None,
                                                                 c_x=c_x
                                                                 )
        combine = T.concatenate([y_emb, next_state, attended], axis=1)

        merge_out = theano.dot(combine, self.W_m) + self.b_m
        if self.max_out:
            merge_out = merge_out.reshape((merge_out.shape[0],
                                           merge_out.shape[1]/2,
                                           2), ndim=3).max(axis=2)
        else:
            merge_out = T.tanh(merge_out)

        return next_state, merge_out


class Translate(object):

    def __init__(self,
                 enc_nhids=1000,
                 dec_nhids=1000,
                 enc_embed=620,
                 dec_embed=620,
                 src_vocab_size=30000,
                 trg_vocab_size=30000,
                 **kwargs):
        self.src_lookup_table = Lookup_table(enc_embed, src_vocab_size, prefix='src_lookup_table')
        self.trg_lookup_table = Lookup_table(dec_embed, trg_vocab_size, prefix='trg_lookup_table')
        self.encoder = BiGRU(enc_embed, enc_nhids, **kwargs)
        self.decoder = Decoder(dec_embed, dec_nhids, c_hids=enc_nhids*2, **kwargs)
        self.logistic = LogisticRegression(kwargs.get('n_out', dec_nhids), trg_vocab_size, prefix='logistic', drop_rate=kwargs['dropout'])
        self.params = self.src_lookup_table.params + self.trg_lookup_table.params + self.encoder.params + self.decoder.params  \
            + self.logistic.params
        self.tparams = OrderedDict([(param.name, param) for param in self.params])

    def apply(self, source, source_mask, target, target_mask, **kwargs):
        sbelow = self.src_lookup_table.apply(source)
        tbelow = self.trg_lookup_table.apply_zero_pad(target)

        s_rep = self.encoder.apply(sbelow, source_mask)
        hiddens = self.decoder.apply(tbelow, target_mask, s_rep, source_mask)

        cost_matrix = self.logistic.cost(hiddens, target, target_mask)
        self.cost = cost_matrix.sum()/target_mask.shape[1]

    def _next_prob_state(self, y, state, c, c_x):
        next_state, merge_out = self.decoder.next_state_merge(y, state, c, c_x)
        prob = self.logistic.apply(merge_out)
        return prob, next_state

    def build_sample(self):
        x = T.matrix('x', dtype='int64')
        sbelow = self.src_lookup_table.apply(x)
        ctx = self.encoder.apply(sbelow, mask=None)
        c_x = T.dot(ctx, self.decoder.Ws) + self.decoder.bs
        init_state = self.decoder.init_state(ctx)

        outs = [init_state, ctx]
        f_init = theano.function([x], outs, name='f_init')

        y = T.vector('y_sampler', dtype='int64')
        y_emb = self.trg_lookup_table.index(y)
        init_state = T.matrix('init_state', dtype='float32')
        next_probs, next_state = self._next_prob_state(y_emb, init_state, ctx, c_x)

        inps = [y, ctx, init_state]
        outs = [next_probs, next_state]
        f_next = theano.function(inps, outs, name='f_next')

        return f_init, f_next

    def savez(self, filename):
        params_value = OrderedDict([(kk, value.get_value()) for kk, value in self.tparams.iteritems()])
        numpy.savez(filename, **params_value)

    def load(self, filename):
        params_value = numpy.load(filename)
        assert len(params_value.files) == len(self.tparams)
        for key, value in self.tparams.iteritems():
            value.set_value(params_value[key])


