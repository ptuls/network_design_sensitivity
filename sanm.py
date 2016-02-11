try:
    import numpy as np
except ImportError:
    raise ImportError("Requires numpy version 1.10 and above")

from random import gauss
from ipf import IPF


__author__ = 'Paul Tune'
__date__ = '11 Feb 2016'


class SANM(object):
    """
    The Spherically Additive Noise Model (SANM) generates a purely spatial traffic matrix
    around a predicted traffic matrix. Require Iterative Proportional Fitting (IPF) from
    ipf.py.
    """
    def __init__(self, predicted):
        """
        Initialization of SANM with the predicted traffic matrix.

        :param predicted: predicted traffic matrix
        """
        if predicted.any() < 0:
            raise ValueError("Predicted traffic matrix must be non-negative")

        self.predicted = np.matrix(predicted)
        self.row_sums = self.predicted.sum(axis=1)
        self.col_sums = self.predicted.sum(axis=0)

    def generate(self, beta, tol=1e-3):
        """
        Generates a single instance of a synthetic traffic matrix based on noise
        parameter beta.

        Example:
        >>print(sanm.generate(0.1))
        [[ 0.19755992  0.40244008]
        [ 0.20244008  0.89755992]]

        :param beta: noise strength parameter
        :param tol: tolerance for IPF's scaling
        :return: a single instance of a synthetic traffic matrix
        """
        tm_size = self.predicted.shape
        tm_generated = np.zeros(tm_size)

        # SANM
        for i in range(tm_generated.shape[0]):
            for j in range(tm_generated.shape[1]):
                tm_generated[i, j] = (np.sqrt(self.predicted[i, j]) + beta*gauss(0, 1))**2

        # run IPF
        IPF().run(tm_generated, self.row_sums, self.col_sums, tol=tol)
        return tm_generated
