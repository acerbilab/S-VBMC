# AI Summary: Synthetic target distributions for testing (GMM & Ring) now inside package namespace.
# AI Summary: Provides synthetic target distributions (GMM & Ring) used for testing and examples in SVBMC.
__all__ = ["GMM", "Ring"]

import numpy as np
from scipy.special import logsumexp


# Defaults as in the paper for the GMM

DEFAULT_MUS = np.array([
    [-8.26197841, -8.75505571], [-6.99244831, -8.41215022],
    [-7.94352608, -7.96066128], [-6.94653653, -6.25663673],
    [-8.31800964, -8.86116925], [-5.35377745,  9.09808750],
    [-6.57295220,  7.03574990], [-6.51023244,  7.15063425],
    [-6.45464607,  6.41256552], [-5.08644145,  8.34791521],
    [ 5.39972878, -5.16531348], [ 6.84830038, -4.83860713],
    [ 6.79133501, -4.86439174], [ 6.04858289, -7.71908624],
    [ 6.67275726, -4.80883357], [ 5.30248146,  4.88622069],
    [ 6.22996358,  4.04107658], [ 4.86277661,  6.56073311],
    [ 4.38009601,  6.32750016], [ 5.11056247,  6.32234096]
])

DEFAULT_SIGMAS = np.array([
    [[1.,  0.5], [0.5, 1.]], [[1.,  0.5], [0.5, 1.]],
    [[1., -0.5], [-0.5, 1.]], [[1., -0.5], [-0.5, 1.]],
    [[1., -0.5], [-0.5, 1.]], [[1., -0.5], [-0.5, 1.]],
    [[1.,  0.5], [0.5, 1.]], [[1.,  0.5], [0.5, 1.]],
    [[1., -0.5], [-0.5, 1.]], [[1.,  0.5], [0.5, 1.]],
    [[1.,  0.5], [0.5, 1.]], [[1., -0.5], [-0.5, 1.]],
    [[1., -0.5], [-0.5, 1.]], [[1.,  0.5], [0.5, 1.]],
    [[1.,  0.5], [0.5, 1.]], [[1., -0.5], [-0.5, 1.]],
    [[1., -0.5], [-0.5, 1.]], [[1., -0.5], [-0.5, 1.]],
    [[1.,  0.5], [0.5, 1.]], [[1., -0.5], [-0.5, 1.]]
])


class GMM:
    """
    Unnormalized multivariate Gaussian mixture. 
    By default it has 20 2D components clustered around 4 centroids.

    Parameters:
    -----------
    mus : np.ndarray or None
        The means of the components. If None, defaults are used.
    sigmas : np.ndarray or None
        The covariance matrices of the components. If None, defaults are used.
    """

    def __init__(self, 
                 mus : np.ndarray | None = None, 
                 sigmas : np.ndarray | None = None):
        
        # component means
        self.mus = mus if mus is not None else DEFAULT_MUS

        # component covariances
        self.sigmas = sigmas if sigmas is not None else DEFAULT_SIGMAS

        # handy constants 
        self.K, self.D = self.mus.shape # Number of components and number of dimensions
        self.inv_sigmas = np.linalg.inv(self.sigmas)          # [`K`, `D`, `D`]
        _, self.logdets = np.linalg.slogdet(self.sigmas)      # [`K`]
        self.log_norm = -0.5 * (self.logdets + self.D * np.log(2 * np.pi))


    def log_pdf(self, x):
        """
        Log-density of the unnormalized mixture at point `x` (shape (`D`,)).
        """
        x = np.asarray(x)
        diff = self.mus - x                                    # [`K`, `D`]
        quad = np.einsum('ki,kij,kj->k', diff, self.inv_sigmas, diff)
        log_components = -0.5 * quad + self.log_norm           # [`K`]
        return logsumexp(log_components)


    def sample(self, n=1):
        """
        Draw `n` samples.  Returns an array of shape (`n`, `D`).
        """
        idx = np.random.randint(0, self.K, size=n)
        return np.array([
            np.random.multivariate_normal(self.mus[i], self.sigmas[i]) for i in idx
        ])


class Ring:
    """
    2-D ring distribution. Defaults as in paper

    Parameters
    ----------
    R: float              
        mean radius
    sigma: float
        radial standard deviation
    center : np.ndarray, float  
        ring center 
    """

    def __init__(self, 
                 R: float = 8, 
                 sigma: float = 0.1, 
                 center=(1., -2.)):
        
        self.R     = R # mean radius
        self.sigma = sigma # standard deviation around radius
        self.center = np.asarray(center, dtype=float)


    # Log-density                                                        
    def log_pdf(self, x : np.ndarray):
        """
        x : (2,) array or (..., 2) array
        Returns an array of the same leading shape as x.
        """
        x = np.asarray(x)
        d  = x - self.center        # vector(s) from center
        r  = np.linalg.norm(d, axis=-1)
        return np.log(r) - 0.5 * ((r - self.R) / self.sigma)**2


    # Sampling                                                           
    def sample(self, n : int = 1):
        """
        Draw `n` samples.  
        Returns an array of shape (`n`, 2).
        """
        samples = np.empty((n, 2))
        i = 0
        rng = np.random

        while i < n:
            # propose a radius
            r = self.R + self.sigma * rng.randn()
            # radius can't be negative
            if r <= 0:
                continue

            # angle + Cartesian projection
            theta = 2.0 * np.pi * rng.rand()
            samples[i] = self.center + r * np.array([np.cos(theta), np.sin(theta)])
            i += 1

        return samples



