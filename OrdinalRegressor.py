from sklearn.base import clone
import numpy as np
from collections import Counter
from OptimizeThresholds import OptimizedRounder

        
class OrdinalRegressor():    
    def __init__(self, clf, **kwargs):
        self.clf = clf(**kwargs)
#         self.dist = None
        self.threshold_optimizer = OptimizedRounder([0,1,2,3])
        
    def fit(self, X, y, **fit_params):
#         self.dist = Counter(y)
#         for k in self.dist:
#             self.dist[k] /= y.size
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
#         acum = 0
#         bound = {}
#         for i in range(3):
#             acum += self.dist[i]
#             bound[i] = np.percentile(pred, acum * 100)
#         # print('y_classify bounds:', bound)
        
#         def classify_example(x):
#             if x <= bound[0]:
#                 return 0
#             elif x <= bound[1]:
#                 return 1
#             elif x <= bound[2]:
#                 return 2
#             else:
#                 return 3
        
#         return list(map(classify_example, pred))
        return self.threshold_optimizer.predict(pred)
    
    def predict_and_classify(self, X):
        return self.classify(self.predict(X))
    
    def predict_proba(self, X):
        return self.predict_and_classify(X)
    
    def decision_function(self, X):
        return self.predict_and_classify(X)