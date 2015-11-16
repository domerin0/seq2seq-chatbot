import theano, numpy
from theano import tensor as T

# nv :: size of our vocabulary
# de :: dimension of the embedding space
# cs :: context window size

def main():
    nv, de, cs, ne = 1000, 50, 5, 500

    emb = theano.shared(name='embeddings', value=0.2 * numpy.random.uniform(-1.0, 1.0, (ne+1, de))
    .astype(theano.config.floatX))

    idxs = T.imatrix()
    x    = emb[idxs].reshape((idxs.shape[0], de*cs))

    sample = [i for i in range(5)]
    csample = contextwin(sample, 7)

    f = theano.function(inputs=[idxs], outputs = x)

    f(csample)

    f(csample).shape

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

if __name__=="__main__":
    main()
