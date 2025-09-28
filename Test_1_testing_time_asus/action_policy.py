from sklearn.base import BaseEstimator, ClassifierMixin

class ActionPolicy(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        pass  # Dynamically set by pickle or code