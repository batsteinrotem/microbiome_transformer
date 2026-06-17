import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Normalizer(BaseEstimator, TransformerMixin):
    """
    Applies Compositional Normalization (Relative Abundance or CLR).
    """
    def __init__(self, method='clr', prevent_zero=0.0001):
        self.method = method
        self.prevent_zero = prevent_zero

    def fit(self, X, y=None):
        # No statistics to learn (both methods are row-wise), but n_features_in_ signals
        # to sklearn that this transformer has been fitted, which Pipeline.transform() requires.
        features_df = X.drop(columns=['sample_id']) if 'sample_id' in X.columns else X.copy()
        self.n_features_in_ = features_df.shape[1]
        return self

    def transform(self, X):
        sample_ids = X['sample_id'] if 'sample_id' in X.columns else None
        features_df = X.drop(columns=['sample_id']) if 'sample_id' in X.columns else X.copy()

        if self.method == 'relative_abundance':
            row_sums = features_df.sum(axis=1)
            row_sums = row_sums.replace(0, 1) # Prevent division by zero
            normalized_df = features_df.div(row_sums, axis=0)

        elif self.method == 'clr':
            features_shifted = features_df + self.prevent_zero
            log_df = np.log(features_shifted)
            # CLR: log(x) - mean(log(x) over all features in the sample)
            normalized_df = log_df.sub(log_df.mean(axis=1), axis=0)
            
        else:
            raise ValueError("Error: unsupported method. Use 'relative_abundance' or 'clr'")

        if sample_ids is not None:
            normalized_df.insert(0, 'sample_id', sample_ids)
            
        return normalized_df