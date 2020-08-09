import numpy as np
import fbm

from .base import Midprice

class FBM(Midprice):
    """Fractional Brownian motion."""
    
    def __init__(self, T=1., H=0.25, sigma=0.02, n=100, drift=0.):
        self.sigma = sigma
        self.n = n
        self.drift = drift
        self.T = T
        self.H = H
        
    def generate(self):
        f = fbm.FBM(n=self.n, hurst=self.H, length=1, method='daviesharte')
        path = np.c_[f.times(), 1 + self.drift * f.times() + self.sigma * f.fbm()]
        
        return path