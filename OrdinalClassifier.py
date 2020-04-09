from sklearn.base import clone
import numpy as np
from sklearn.utils import class_weight

### balance class weights
def sample_balancer(y):
    class_weights = list(class_weight.compute_class_weight('balanced',
                                                 np.unique(y),
                                                 y))
    w_array = np.ones(y.shape[0], dtype = 'float')
    for i, val in enumerate(y):
        w_array[i] = class_weights[val] 
    return w_array

def class_balancer(y):
    classes = np.unique(y)
    class_weights = list(class_weight.compute_class_weight('balanced',
                                                 classes,
                                                 y))

    return {k: v for k,v in zip(classes, class_weights)}
        
class OrdinalClassifier():    
    def __init__(self, clf, balance=None, **kwargs):
        balancers = {
            'sample_weight': sample_balancer,
            'class_weight': class_balancer
        }
        
        self.clf = clf(**kwargs)
        self.clfs = {}
        self.balance_arg = balance if balance in balancers else None
        self.balancer = balancers.get(balance)
        
    def fit(self, X, y, **fit_params):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                
                if self.balancer:
                    w_array = self.balancer(y)
                    fit_params.update({self.balance_arg: w_array})

                clf = clone(self.clf)
                clf.fit(X, binary_y, **fit_params)
                self.clfs[i] = clf
    
    def predict_proba(self, X):
        clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i,y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X, **predict_params):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def set_params(self, **kwargs):
        self.clf = self.clf.set_params(**kwargs)