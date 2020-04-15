from sklearn.base import clone
import numpy as np
from collections import Counter
from OptimizeThresholds import OptimizedRounder

        
class OrdinalRegressor():    
    def __init__(self, clf, **kwargs):
        self.clf = clf(**kwargs)
        self.threshold_optimizer = OptimizedRounder([0,1,2,3])
        
    def fit(self, X, y, **fit_params):
        self.clf.fit(X, y, **fit_params)
        self.threshold_optimizer.fit(self.predict(X), y)
    
    def predict(self, X, **predict_params):
        pred = self.clf.predict(X)
        if predict_params.get('classify'):
            return self.classify(pred)
        return pred
    
    def set_params(self, **kwargs):
        self.clf = self.clf.set_params(**kwargs)
        
    def classify(self, pred):
        return self.threshold_optimizer.predict(pred)
    
    def predict_and_classify(self, X):
        return self.classify(self.predict(X))
    
    def predict_proba(self, X):
        # overload predict_proba method to output the final classification
        return self.predict_and_classify(X)