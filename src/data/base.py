import numpy as np
from tqdm.auto import tqdm
import iisignature
from joblib import Parallel, delayed

import utils
import tensor_algebra as ta


class Midprice(object):
    """Base class for data (i.e. midprice models)."""

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _sig(path, order):
        return np.r_[1., iisignature.sig(utils.transform(path), order)]


    def _generate(self, seed):
        np.random.seed(seed)

        return self.generate()

    def generate(self):
        """Generate a sample path."""

        raise NotImplementedError("Generator not implemented")


    def build(self, n_paths=1000, order=6):
        """Builds paths and ES."""

        # Create paths
        paths = Parallel(n_jobs=-1)(delayed(self._generate)(seed) \
                                    for seed in tqdm(range(n_paths), desc="Building paths"))



        # Compute signatures
        sigs = Parallel(n_jobs=-1)(delayed(self._sig)(path, order) \
                                   for path in tqdm(paths, desc="Computing signatures"))

        # Compute ES
        ES = ta.Tensor(np.mean(sigs, axis=0), 2, order)

        return np.array(paths), ES
