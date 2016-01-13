import theano
import theano.tensor as T
import numpy
from utils import param_init, repeat_x


def _p(pp, name):
    return '%s_%s' % (pp, name)


class LogisticRegression(object):

    def __init__(self, n_in, n_out, prefix='logist'):

        self.n_in = n_in
        self.n_out = n_out
        self.W = param_init().param((n_in, n_out), name=_p(prefix, 'W'))
        self.b = param_init().param((n_out, ), name=_p(prefix, 'b'))
        self.params = [self.W, self.b]

    def apply(self, input):
        energy = theano.dot(input, self.W) + self.b
        if energy.ndim == 3:
            energy = energy.reshape([energy.shape[0]*energy.shape[1], energy.shape[2]])
        pmf = T.nnet.softmax(energy)

        self.p_y_given_x = pmf

        return self.p_y_given_x

    def cost(self, input, targets, mask=None):
        prediction = self.apply(input)

        targets_flat = targets.flatten()

        x_flat_idx = T.arange(targets_flat.shape[0])
        ce = -T.log(prediction[x_flat_idx, targets_flat])
        if mask is None:
            ce = ce.reshape([targets.shape[0], targets.shape[1]])
        else:
            ce = ce.reshape([targets.shape[0], targets.shape[1]]) * mask
        return ce

    def errors(self, y):
        y_pred = T.argmax(self.p_y_given_x, axis=-1)
        if y.ndim == 2:
            y = y.flatten()
            y_pred = y_pred.flatten()
        return T.sum(T.neq(y, y_pred))


class GRU(object):

    def __init__(self, n_in, n_hids, with_contex=False, merge=True, max_out=True, prefix='GRU', **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self.prefix = prefix
        self.merge = merge
        self.max_out = max_out

        self._init_params()

    def _init_params(self):

        f = lambda name: _p(self.prefix, name)

        n_in = self.n_in
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)
        size_hh = (n_hids, n_hids)
        self.W_xz = param_init().param(size_xh, name=f('W_xz'))
        self.W_xr = param_init().param(size_xh, name=f('W_xr'))
        self.W_xh = param_init().param(size_xh, name=f('W_xh'))

        self.W_hz = param_init().param(size_hh, name=f('W_hz'))
        self.W_hr = param_init().param(size_hh, name=f('W_hr'))
        self.W_hh = param_init().param(size_hh, name=f('W_hh'))

        self.b_z = param_init().param((n_hids,), name=f('b_z'))
        self.b_r = param_init().param((n_hids,), name=f('b_r'))
        self.b_h = param_init().param((n_hids,), name=f('b_h'))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        if self.with_contex:
            size_ch = (self.c_hids, self.n_hids)
            self.W_cz = param_init().param(size_ch, name=f('W_cz'))
            self.W_cr = param_init().param(size_ch, name=f('W_cr'))
            self.W_ch = param_init().param(size_ch, name=f('W_ch'))
            self.W_c_init = param_init().param(size_ch, name=f('W_c_init'))

            self.params = self.params + [self.W_cz, self.W_cr,
                                         self.W_ch, self.W_c_init]

            msize = self.n_in + self.n_hids + self.c_hids
        else:
            msize = self.n_in + self.n_hids

        if self.merge:
            osize = self.n_hids
            if self.max_out:
                self.W_m = param_init().param((msize, osize*2), name=_p(self.prefix, 'W_m'))
                self.b_m = param_init().param((osize*2,), name=_p(self.prefix, 'b_m'))
                self.params += [self.W_m, self.b_m]
            else:
                self.W_m = param_init().param((msize, osize*2), name=_p(self.prefix, 'W_m'))
                self.b_m = param_init().param((osize*2,), name=_p(self.prefix, 'b_m'))
                self.params += [self.W_m, self.b_m]



    def _step_forward_with_context(self, x_t, x_m, h_tm1, c_z, c_r, c_h):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        '''
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                             T.dot(h_tm1, self.W_hz) + c_z + self.b_z)

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                             T.dot(h_tm1, self.W_hr) + c_r + self.b_r)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) +
                         c_h + self.b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        if x_m is not None:
            h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t


    def _step_forward(self, x_t, x_m, h_tm1):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        '''
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                             T.dot(h_tm1, self.W_hz) + self.b_z)

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                             T.dot(h_tm1, self.W_hr) + self.b_r)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) +
                         self.b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        if x_m is not None:
            h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t

    def _forward(self, state_below, mask_below=None, init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError

        if mask_below:
            inps = [state_below, mask_below]
            if self.with_contex:
                fn = self._step_forward_with_context
            else:
                fn = self._step_forward
        else:
            inps = [state_below]
            if self.with_contex:
                fn = lambda x1, x2, x3, x4, x5: self._step_forward_with_context(x1, None, x2, x3, x4, x5)
            else:
                fn = lambda x1, x2: self._step_forward(x1, None, x2)

        if self.with_contex:
            if init_state is None:
                init_state = T.tanh(theano.dot(context, self.W_c_init))
            c_z = theano.dot(context, self.W_cz)
            c_r = theano.dot(context, self.W_cr)
            c_h = theano.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]
            rval, updates = theano.scan(fn,
                                        sequences=inps,
                                        outputs_info=[init_state],
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        else:
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                #init_state = T.unbroadcast(T.alloc(0., batch_size, self.n_hids), 0)
            rval, updates = theano.scan(fn,
                                        sequences=inps,
                                        outputs_info=[init_state],
                                        n_steps=n_steps
                                        )
        self.output = rval
        return self.output

    def _merge_out(self, state_below, mask_below=None, context=None):
        hiddens = self._forward(state_below, mask_below=None, context=context)
        if self.with_contex:
            assert context is not None
            n_times = state_below.shape[0]
            m_context = repeat_x(context, n_times)
            combine = T.concatenate([state_below, hiddens, m_context], axis=2)
        else:
            combine = T.concatenate([state_below, hiddens], axis=2)

        if self.max_out:
            merge_out = theano.dot(combine, self.W_m) + self.b_m
            merge_out = merge_out.reshape((merge_out.shape[0],
                                           merge_out.shape[1],
                                           merge_out.shape[2]/2,
                                           2), ndim=4).max(axis=3)
        else:
            merge_out = T.tanh(theano.dot(combine, self.W_m) + self.b_m)
        if mask_below:
            merge_out = merge_out * mask_below[:, :, None]
        return merge_out

    def apply(self, state_below, mask_below, context=None):
        if self.merge:
            return self._merge_out(state_below, mask_below, context)
        else:
            return self._forward(state_below, mask_below, context)



class Lookup_table(object):
    def __init__(self, embsize, vocab_size, prefix='Lookup_table'):
        self.W = param_init().param((vocab_size, embsize), name=_p(prefix, 'embed'))
        self.params = [self.W]
        self.vocab_size = vocab_size
        self.embsize = embsize

    def apply(self, indices):
        outshape = [indices.shape[i] for i
                    in range(indices.ndim)] + [self.embsize]

        return self.W[indices.flatten()].reshape(outshape)

    def index(self, i):
        return self.W[i]


class BiGRU(object):

    def __init__(self, n_in, n_hids, with_contex=False, prefix='BiGRU', **kwargs):
        kwargs['merge'] = False
        self.encoder = GRU(n_in, n_hids, with_contex=with_contex, prefix=_p(prefix, 'l2r'), **kwargs)
        self.rencoder = GRU(n_in, n_hids, with_contex=with_contex, prefix=_p(prefix, 'r2l'), **kwargs)

        self.params = self.encoder.params + self.rencoder.params

    def apply(self, state_below, mask):
        rstate_below = state_below[::-1]
        if mask is None:
            rmask = None
        else:
            rmask = mask[::-1]
        loutput = self.encoder.apply(state_below, mask)
        routput = self.rencoder.apply(rstate_below, rmask)

        self.output = T.concatenate([loutput, routput[::-1]], axis=2)
        return self.output

