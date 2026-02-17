import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TempInteractions(BaseEstimator, TransformerMixin):
    """Temperature interaction features transformer"""
    # ...existing TempInteractions code...

class AdaptiveFeatureScaler(BaseEstimator, TransformerMixin):
    """Adaptive feature scaling transformer"""
    # ...existing AdaptiveFeatureScaler code...

class LearnableFeatureWeights(BaseEstimator, TransformerMixin):
    """Learnable feature weights transformer"""
    # ...existing LearnableFeatureWeights code...