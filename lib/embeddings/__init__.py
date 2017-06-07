'''
Library for reading word vector files (text or binary)
@Python3
'''
import numpy
import codecs
from ..logging import log

from .common import *
from . import word2vec
from . import glove
from .glove import GloveMode

def read(fname, format=Format.Word2Vec, **kwargs):
    '''Returns array of words and word embedding matrix
    '''
    if format == Format.Word2Vec:
        (words, vectors) = word2vec.read(fname, **kwargs)
    elif format == Format.Glove:
        (words, vectors) = glove.read(fname, **kwargs)

    wordmap = {}
    for i in range(len(words)):
        wordmap[words[i]] = vectors[i]
    return wordmap

def load(*args, **kwargs):
    '''Alias for read'''
    return read(*args, **kwargs)

def write(embeds, fname, format=Format.Word2Vec, **kwargs):
    '''Writes a dictionary of embeddings { term : embed}
    to a file, in the format specified.
    '''
    if format == Format.Word2Vec:
        word2vec.write(embeds, fname, **kwargs)
    else:
        return NotImplemented

def readVocab(fname, format=Format.Word2Vec, **kwargs):
    '''Gets the set of words embedded in the given file
    '''
    embs = read(fname, format=format, **kwargs)
    return set(embs.keys())

def listVocab(embeds, fname):
    '''Writes the vocabulary of an embedding to a file,
    one entry per line.
    '''
    with open(fname, 'wb') as stream:
        for k in embeds.keys():
            stream.write(k.encode('utf-8'))
            stream.write(b'\n')

def closestNeighbor(query, embedding_array, normed=False, top_k=1):
    '''Gets the index of the closest neighbor of embedding_array
    to the query point.  Distance metric is cosine.

    SLOW. DO NOT USE THIS FOR RAPID COMPUTATION.
    '''
    embedding_array = numpy.array(embedding_array)
    if not normed:
        embedding_array = numpy.array([
            (embedding_array[i] / numpy.linalg.norm(embedding_array[i]))
                for i in range(embedding_array.shape[0])
        ])

    ## assuming embeddings are unit-normed by this point;
    ## norm(query) is a constant factor, so we can ignore it
    dists = numpy.array([
        numpy.dot(query, embedding_array[i])
            for i in range(embedding_array.shape[0])
    ])
    sorted_ixes = numpy.argsort(-1 * dists)
    return sorted_ixes[:top_k]

def unitNorm(embeds):
    for (k, embed) in embeds.items():
        embeds[k] = numpy.array(
            embed / numpy.linalg.norm(embed)
        )

def analogyQuery(embeds, a, b, c):
    return (
        numpy.array(embeds[b])
        - numpy.array(embeds[a])
        + numpy.array(embeds[c])
    )

def splitVocabAndEmbeddings(embeds):
    vocab = tuple(embeds.keys())
    embed_array = []
    for v in vocab: embed_array.append(embeds[v])
    return (vocab, embed_array)


class NearestNeighbors:
    '''Used to get nearest embeddings to a query by cosine distance.
    '''
    
    def __init__(self, embeds):
        e = embeds.copy()
        unitNorm(e)

        vocab = tuple(embeds.keys())
        embed_array = []
        for v in vocab:
            embed_array.append(embeds[v])
        self._vocab = numpy.array(vocab)
        self._embed_array = numpy.transpose(numpy.array(embed_array))

    def nearest(self, query, k=1):
        indices = numpy.argsort(
            numpy.matmul(
                numpy.array(query),
                self._embed_array
            )
        )
        rev_sort_keys = self._vocab[indices][::-1]
        if k is None: return rev_sort_keys
        else: return rev_sort_keys[1:k+1]
