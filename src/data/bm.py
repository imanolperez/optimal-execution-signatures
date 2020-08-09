import numpy as np
from .base import Midprice

class BM(Midprice):
    """Brownian motion."""
    
    def __init__(self, T=1., sigma=0.02, n=100, drift=0.):
        self.sigma = sigma
        self.n = n
        self.drift = drift
        self.T = T
        
    def generate(self):
        dt = self.sigma ** 2 * self.T / self.n
        
        path = 1. + np.r_[[0.], np.sqrt(dt) * np.random.randn(self.n - 1).cumsum()]
        path = np.c_[np.linspace(0, self.T, self.n), path]
        path[:, 1] += self.drift * path[:, 0]
        
        return path