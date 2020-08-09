import numpy as np
import itertools
import torch


class Tensor(object):

    def __init__(self, sig, dim, order):
        self.dim = dim
        self.order = order

        self.value = [torch.zeros([dim] * n) for n in range(order + 1)]

        keys = list(self.sigkeys())

        for val, key in zip(sig, keys):
            self[key] = val
        


    def __getitem__(self, key):
        assert isinstance(key, tuple), "key is not a tuple"
        return self.value[len(key)][key]

    def __setitem__(self, key, value):
        assert isinstance(key, tuple), "key is not a tuple"

        self.value[len(key)][key] = value

    def sigkeys(self):
        for n in range(self.order + 1):
            for w in itertools.product(*[list(range(self.dim)) for _ in range(n)]):
                yield w

    def flatten(self):
        v = []
        for val in self.value:
            v = np.r_[v, val.detach().numpy().flatten()]
        return v

