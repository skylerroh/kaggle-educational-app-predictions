import pandas as pd
from functools import partial
from sklearn.metrics import cohen_kappa_score
import scipy as sp
import numpy as np

class OptimizedRounder(object):
    def __init__(self, labels):
        self.coef_ = 0
        self.labels = labels
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels)
        return -cohen_kappa_score(y, preds, weights = 'quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')
    
    def predict(self, X, coef=None):
        coef = coef if coef else self.coefficients()
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels)
        return preds
    
    def coefficients(self):
        return self.coef_['x']