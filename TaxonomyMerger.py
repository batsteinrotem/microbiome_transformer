import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class TaxonomyMerger:
    def __init__(self, method='sum'):
        self.method = method
        self.pca_models = {}
        self.duplicated_names = []
        self.unique_cols_mask = None

    def fit_transform(self, df_train):
        """
        This function conducts identical columns merge on the data
        """
        sample_ids = df_train['sample_id']
        features_df = df_train.drop(columns=['sample_id'])
        
        # Return original data if there are no duplications
        if not features_df.columns.duplicated().any():
            return df_train

        if self.method in ['sum', 'mean']:
            # Applies sum or mean on all data 
            return self.transform(df_train)
            
        elif self.method == 'pca':
            # Get names of duplicate columns 
            self.unique_cols_mask = ~features_df.columns.duplicated(keep=False)
            # Apply boolean masking to get duplicated columns
            processed_df = features_df.loc[:, self.unique_cols_mask].copy()
            self.duplicated_names = features_df.columns[features_df.columns.duplicated()].unique()
            
            pca_results = {}
            # For each column perform PCA 
            for col_name in self.duplicated_names:
                sub_df = features_df[[col_name]].fillna(0)
                pca = PCA(n_components=1)
                
                # Fit the PCA to the train data and transform it
                pc1 = pca.fit_transform(sub_df).ravel()
                
                # Add min value to ensure CLR competability 
                min_val = np.min(pc1)
                pc1_shifted = pc1 - min_val + 1e-6
                
                # Save the PCA model and min value for test data transformation
                self.pca_models[col_name] = {'model': pca, 'min_val': min_val}
                pca_results[col_name] = pc1_shifted
            
            if pca_results:
                pca_df = pd.DataFrame(pca_results, index=features_df.index)
                processed_df = pd.concat([processed_df, pca_df], axis=1)
                
            processed_df.insert(0, 'sample_id', sample_ids)
            return processed_df
        else:
            raise ValueError("Error: unsupported method")

    def transform(self, df_new):
        """
        transform the data (test, validation) without PCA fit
        """
        if self.unique_cols_mask is None:
            return df_new
        sample_ids = df_new['sample_id']
        features_df = df_new.drop(columns=['sample_id'])

        if self.method == 'sum':
            processed_df = features_df.groupby(features_df.columns, axis=1).sum()
        elif self.method == 'mean':
            processed_df = features_df.groupby(features_df.columns, axis=1).mean()
        elif self.method == 'pca':
            # Use the Boolean Masking from the train data
            processed_df = features_df.loc[:, self.unique_cols_mask].copy()
            pca_results = {}
            
            # for each column of the duplicated once perform merger
            for col_name in self.duplicated_names:
                sub_df = features_df[[col_name]].fillna(0)
                
                # Get the PCA model from the train data
                pca_info = self.pca_models[col_name]
                pca_model = pca_info['model']
                train_min_val = pca_info['min_val']
                
                # Apply transformation on the test data
                pc1 = pca_model.transform(sub_df).ravel()
                
                # Shift the values using the min val calculated from train data
                pc1_shifted = pc1 - train_min_val + 1e-6
                # If there are still negative numbers replace them with min value of 1e-6
                pc1_shifted = np.clip(pc1_shifted, a_min=1e-6, a_max=None) 
                
                pca_results[col_name] = pc1_shifted
                
            if pca_results:
                pca_df = pd.DataFrame(pca_results, index=features_df.index)
                processed_df = pd.concat([processed_df, pca_df], axis=1)
                
        processed_df.insert(0, 'sample_id', sample_ids)
        return processed_df
    
