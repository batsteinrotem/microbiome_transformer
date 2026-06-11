import numpy as np

class Normalizer:
    def __init__(self, method='relative_abundance', pseudo_count=1e-6):
        """
        Normalize the data using one of the selected methods: relative_abundance, CLR
        """
        self.method = method
        self.pseudo_count = pseudo_count

    def fit(self, df):
        # fit function for compatibility with Scikit-Learn Pipelines.
        return self

    def transform(self, df):
        """
        Apply data normalization
        """
        sample_ids = df['sample_id']
        features_df = df.drop(columns=['sample_id']).copy()

        # Normalize using relative abundance
        if self.method == 'relative_abundance':
            # sum all values of the sample
            row_sums = features_df.sum(axis=1)
            row_sums[row_sums == 0] = 1 
            
            # divide each value by the sum for each row
            normalized_df = features_df.div(row_sums, axis=0)

        elif self.method == 'clr':
            # Add pseudo value to each cell to enable log transformation
            features_shifted = features_df + self.pseudo_count
            
            # Calculate log of each value
            log_df = np.log(features_shifted)
            
            # Calculate the log mean for each row and divide by the log of each cell value
            normalized_df = log_df.sub(log_df.mean(axis=1), axis=0)

        else:
            raise ValueError("Error: unsupported method. Use 'relative_abundance' or 'clr'")

        normalized_df.insert(0, 'sample_id', sample_ids)
        return normalized_df

    def fit_transform(self, df):
        return self.fit(df).transform(df)