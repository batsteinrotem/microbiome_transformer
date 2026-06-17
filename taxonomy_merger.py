import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class TaxonomyMerger(BaseEstimator, TransformerMixin):
    """
    Merges duplicate taxonomic columns using Sum, Mean, or PCA.
    Separate fit (learning parameters) from transform (applying them).
    """
    def __init__(self, method='pca'):
        self.method = method
        self.pca_models = {}
        self.duplicated_names = []
        self.unique_col_names = []
        self._is_fitted = False

    # The fit function builds a PCA model on the training data 
    def fit(self, X, y=None):
        features_df = X.drop(columns=['sample_id']) if 'sample_id' in X.columns else X.copy()
        # n_features_in_ follows sklearn's convention for marking an estimator as fitted
        self.n_features_in_ = features_df.shape[1]

        if not features_df.columns.duplicated().any():
            self._is_fitted = True
            return self

        # Save explicit names for duplicated and unique columns
        unique_mask = ~features_df.columns.duplicated(keep=False)
        self.unique_col_names = features_df.columns[unique_mask].tolist()
        self.duplicated_names = features_df.columns[features_df.columns.duplicated()].unique().tolist()

        if self.method == 'pca':
            # Learn PCA weights and min values ONLY from training data
            for col_name in self.duplicated_names:
                sub_df = features_df[[col_name]].fillna(0)
                pca = PCA(n_components=1)
                # Pass .values (numpy) to remove column names, ravel to flatten the 2d array
                pc1_train = pca.fit_transform(sub_df.values).ravel()
                
                # Save the trained model and the minimum value for shifting later
                self.pca_models[col_name] = {
                    'model': pca,
                    'min_val': np.min(pc1_train)
                }
                
        self._is_fitted = True
        return self

    def transform(self, X):
        # If the PCA is not fitted on the training data raise error
        if not self._is_fitted:
            raise RuntimeError("You must fit before transforming data!")
            
        sample_ids = X['sample_id'] if 'sample_id' in X.columns else None
        features_df = X.drop(columns=['sample_id']) if 'sample_id' in X.columns else X.copy()
        
        # REMOVED: The early return checking for incoming duplicated columns.
        # We must transform based on what was learned during 'fit', regardless of the incoming batch shape.

        # If fit found no duplicates initially, just return the data
        if not self.duplicated_names:
            return X.copy()

        if self.method == 'sum':
            processed_df = features_df.T.groupby(level=0).sum().T
        elif self.method == 'mean':
            processed_df = features_df.T.groupby(level=0).mean().T
        elif self.method == 'pca':
            # Safely extract unique columns by explicit name
            processed_df = features_df[self.unique_col_names].copy()
            pca_results = {}

            # Apply the pre-trained PCA models to the incoming data
            for col_name in self.duplicated_names:
                # Fallback in case a duplicate column is missing
                if col_name not in features_df.columns:
                    continue

                sub_df = features_df[[col_name]].fillna(0)
                pca_info = self.pca_models[col_name]

                # Pass .values (numpy) to avoid sklearn/narwhals duplicate-column validation
                pc1 = pca_info['model'].transform(sub_df.values).ravel()
                
                # Shift using the TRAINED minimum value to prevent negative numbers
                pc1_shifted = pc1 - pca_info['min_val']
                pc1_shifted = np.clip(pc1_shifted, a_min=0.0, a_max=None)
                
                pca_results[col_name] = pc1_shifted

            # Combine the unique columns df and the new PCA df    
            if pca_results:
                pca_df = pd.DataFrame(pca_results, index=features_df.index)
                processed_df = pd.concat([processed_df, pca_df], axis=1)
        else:
            raise ValueError(f"Method '{self.method}' is not supported.")

        if sample_ids is not None:
            processed_df.insert(0, 'sample_id', sample_ids)
            
        return processed_df