'''
Abstracted access to embedding for a target phrase; handles word-level backoff under the hood.
'''

import numpy as np

class EmbeddingWrapper:

    _embeds = None
    _embed_vocab = None
    _embed_array = None
    _backoff_embeds = None
    
    def __init__(self, embeds, backoff_embeds=None):
        # for indexed access, split embeddings dictionary into immutable vocabulary and array
        self._embed_vocab = tuple(embeds.keys())
        self._embed_vocab_indices = { self._embed_vocab[i]:i for i in range(len(self._embed_vocab)) }

        self._embeds = embeds
        self._backoff_embeds = backoff_embeds

    def index(self, item):
        try:
            return self._embed_vocab_indices[item]
        except (KeyError, ValueError):
            return -1

    def indexToTerm(self, ix):
        return self._embed_vocab[ix]

    def asArray(self):
        return np.array([self._embeds[v] for v in self._embed_vocab])

    def __getitem__(self, item):
        if not type(item) is int:
            # use pre-calculated phrase embedding if known, else back off to averaging known words
            if self._embeds.get(item, None) != None: return self._embeds[item]
            else:
                return self._laxTokenAverage(item, self._backoff_embeds)
        else:
            return self._embeds[self._embed_vocab[item]]

    def _laxTokenAverage(self, item, embeds):
        token_embeds = []
        for t in item.split():
            if not embeds.get(t, None) is None: token_embeds.append(embeds[t])
        if len(token_embeds) > 0:
            return np.mean(token_embeds, axis=0)
        else: raise KeyError
