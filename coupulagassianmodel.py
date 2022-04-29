import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from typing import List,Callable
import torch

class MCGM:
    
    def __init__(self, K:int,marginals:List[Callable[[float],float]]) -> None:
        self.K = K
        self.marginals=marginals

    @staticmethod
    def GC(x,mu,sigma):
        u = np.array([norm.ppf(self.marginas[])])

    def fit(self,X):
        d=X.shape[1] # number of dimensions
        m=X.shape[0] # number of samples

        mus = torch.empty(self.K,d,1)
        sigmas = np.empty(self.K,d,d)

        gaussians = [multivariate_normal(mus[k],sigmas[k]) for k in range(self.K)]





