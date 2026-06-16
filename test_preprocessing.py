import sys
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TaxonomyMerger import TaxonomyMerger
from Normalize import Normalizer


# ── Shared test-data factories ────────────────────────────────────────────────

def _df_no_dups(n=20, seed=0):
    """DataFrame with no duplicate columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'sample_id': [f'S{i:03d}' for i in range(n)],
        'taxon_A':   rng.random(n),
        'taxon_B':   rng.random(n),
        'taxon_C':   rng.random(n),
    })


def _df_with_dups(n=30, seed=0):
    """DataFrame where taxon_A and taxon_B each appear TWICE."""
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({
        'sample_id': [f'S{i:03d}' for i in range(n)],
        'taxon_A':   rng.random(n),
        'taxon_B':   rng.random(n),
        'taxon_C':   rng.random(n),
    })
    extra = pd.DataFrame({
        'taxon_A': rng.random(n),
        'taxon_B': rng.random(n),
    })
    return pd.concat([base, extra], axis=1)


def _df_labeled(n=100, seed=42):
    """Full DataFrame with study_condition labels for split tests."""
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        'sample_id':       [f'S{i:04d}' for i in range(n)],
        'taxon_A':         rng.random(n),
        'taxon_B':         rng.random(n),
        'study_condition': ['CRC'] * half + ['healthy'] * (n - half),
    })


# ── TaxonomyMerger – no duplicates ────────────────────────────────────────────

class TestTaxonomyMergerNoDuplicates(unittest.TestCase):
    """When the input has no duplicate columns the data must be returned unchanged."""

    def setUp(self):
        self.df = _df_no_dups()

    def _passthrough(self, method):
        merger = TaxonomyMerger(method=method)
        result = merger.fit_transform(self.df)
        pd.testing.assert_frame_equal(result, self.df)

    def test_sum_passthrough(self):   self._passthrough('sum')
    def test_mean_passthrough(self):  self._passthrough('mean')
    def test_pca_passthrough(self):   self._passthrough('pca')

    def test_column_count_unchanged(self):
        result = TaxonomyMerger(method='sum').fit_transform(self.df)
        self.assertEqual(result.shape[1], self.df.shape[1])


# ── TaxonomyMerger – sum ──────────────────────────────────────────────────────

class TestTaxonomyMergerSum(unittest.TestCase):

    def setUp(self):
        # Controlled: two 'taxon_A' columns with known values; one 'taxon_B'
        base = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'taxon_A':   [1.0, 2.0, 3.0],
            'taxon_B':   [10.0, 20.0, 30.0],
        })
        extra = pd.DataFrame({'taxon_A': [4.0, 5.0, 6.0]})
        self.df = pd.concat([base, extra], axis=1)

    def test_column_count_after_merge(self):
        result = TaxonomyMerger(method='sum').fit_transform(self.df)
        self.assertEqual(result.shape[1], 3)   # sample_id + taxon_A + taxon_B

    def test_sum_values_correct(self):
        result = TaxonomyMerger(method='sum').fit_transform(self.df)
        np.testing.assert_array_almost_equal(
            result['taxon_A'].values, [1+4, 2+5, 3+6])

    def test_non_duplicate_column_unchanged(self):
        result = TaxonomyMerger(method='sum').fit_transform(self.df)
        np.testing.assert_array_equal(result['taxon_B'].values, [10.0, 20.0, 30.0])

    def test_sample_id_preserved(self):
        result = TaxonomyMerger(method='sum').fit_transform(self.df)
        self.assertListEqual(result['sample_id'].tolist(), ['S1', 'S2', 'S3'])

    def test_no_duplicate_cols_in_output(self):
        result = TaxonomyMerger(method='sum').fit_transform(self.df)
        self.assertFalse(result.columns.duplicated().any())

    def test_transform_consistent_with_fit_transform(self):
        """transform() on the same data should produce the same result as fit_transform()."""
        merger = TaxonomyMerger(method='sum')
        ft_result = merger.fit_transform(self.df)
        t_result  = TaxonomyMerger(method='sum').fit(self.df).transform(self.df)
        pd.testing.assert_frame_equal(ft_result, t_result)


# ── TaxonomyMerger – mean ─────────────────────────────────────────────────────

class TestTaxonomyMergerMean(unittest.TestCase):

    def setUp(self):
        base  = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'taxon_A':   [2.0, 4.0, 6.0],
        })
        extra = pd.DataFrame({'taxon_A': [4.0, 8.0, 12.0]})
        self.df = pd.concat([base, extra], axis=1)

    def test_mean_values_correct(self):
        result = TaxonomyMerger(method='mean').fit_transform(self.df)
        expected = [(2+4)/2, (4+8)/2, (6+12)/2]
        np.testing.assert_array_almost_equal(result['taxon_A'].values, expected)

    def test_reduces_to_unique_cols(self):
        result = TaxonomyMerger(method='mean').fit_transform(self.df)
        self.assertEqual(result.shape[1], 2)   # sample_id + taxon_A

    def test_no_duplicate_cols_in_output(self):
        result = TaxonomyMerger(method='mean').fit_transform(self.df)
        self.assertFalse(result.columns.duplicated().any())


# ── TaxonomyMerger – PCA ──────────────────────────────────────────────────────

class TestTaxonomyMergerPCA(unittest.TestCase):

    def setUp(self):
        self.df = _df_with_dups(n=30, seed=1)

    def test_column_count_reduced(self):
        result = TaxonomyMerger(method='pca').fit_transform(self.df)
        n_in  = self.df.shape[1] - 1          # feature cols (no sample_id)
        n_out = result.shape[1] - 1
        self.assertLess(n_out, n_in,
            "Merging duplicate columns must reduce the feature count")

    def test_no_duplicate_cols_in_output(self):
        result = TaxonomyMerger(method='pca').fit_transform(self.df)
        self.assertFalse(result.columns.duplicated().any())

    def test_pca_output_nonnegative(self):
        result = TaxonomyMerger(method='pca').fit_transform(self.df)
        feature_cols = [c for c in result.columns if c != 'sample_id']
        self.assertTrue((result[feature_cols].values >= 0).all(),
            "Shifted & clipped PCA values must be non-negative")

    def test_sample_id_preserved(self):
        result = TaxonomyMerger(method='pca').fit_transform(self.df)
        self.assertIn('sample_id', result.columns)
        self.assertListEqual(
            result['sample_id'].tolist(),
            self.df['sample_id'].tolist()
        )

    def test_no_data_leakage_pca_params_from_train(self):
        """
        PCA min_val is learned on train only.
        Calling transform() on test data must NOT update the fitted models.
        """
        train = _df_with_dups(n=30, seed=1)
        test  = _df_with_dups(n=10, seed=99)

        merger = TaxonomyMerger(method='pca')
        merger.fit(train)

        min_before = merger.pca_models['taxon_A']['min_val']
        merger.transform(test)
        min_after  = merger.pca_models['taxon_A']['min_val']

        self.assertEqual(min_before, min_after,
            "transform() must not refit or modify stored PCA parameters")

    def test_train_and_test_pca_params_differ(self):
        """
        Sanity-check that train and test distributions are actually different,
        which is the precondition for the leakage test above to be meaningful.
        """
        train = _df_with_dups(n=30, seed=1)
        test  = _df_with_dups(n=30, seed=99)

        m_train = TaxonomyMerger(method='pca').fit(train)
        m_test  = TaxonomyMerger(method='pca').fit(test)

        self.assertNotAlmostEqual(
            m_train.pca_models['taxon_A']['min_val'],
            m_test.pca_models['taxon_A']['min_val'],
            places=5,
            msg="Train and test should have different PCA min_vals (different seeds)"
        )

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            TaxonomyMerger(method='pca').transform(self.df)

    def test_unsupported_method_raises(self):
        merger = TaxonomyMerger(method='bad_method')
        merger._is_fitted   = True
        merger.duplicated_names = ['taxon_A']
        with self.assertRaises(ValueError):
            merger.transform(_df_with_dups())


# ── Normalizer – relative abundance ──────────────────────────────────────────

class TestNormalizerRelativeAbundance(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'sample_id': ['S1', 'S2', 'S3'],
            'a': [1.0, 0.0, 3.0],
            'b': [3.0, 0.0, 1.0],
            'c': [6.0, 0.0, 6.0],
        })

    def _feature_cols(self, df):
        return [c for c in df.columns if c != 'sample_id']

    def test_rows_sum_to_one(self):
        result = Normalizer(method='relative_abundance').fit_transform(self.df)
        row_sums = result[self._feature_cols(result)].sum(axis=1)
        # S2 is all-zero — correct output is all-zero (no meaningful normalization)
        np.testing.assert_array_almost_equal(row_sums.values, [1.0, 0.0, 1.0])

    def test_all_zero_row_no_nan_no_inf(self):
        """Row S2 is all zeros – division by zero must be handled gracefully."""
        result = Normalizer(method='relative_abundance').fit_transform(self.df)
        self.assertFalse(result.isnull().any().any(), "No NaN expected")
        numeric = result[self._feature_cols(result)].values
        self.assertFalse(np.isinf(numeric).any(), "No inf expected")

    def test_values_between_zero_and_one(self):
        result = Normalizer(method='relative_abundance').fit_transform(self.df)
        vals = result[self._feature_cols(result)].values
        self.assertTrue((vals >= 0).all() and (vals <= 1).all())

    def test_sample_id_preserved(self):
        result = Normalizer(method='relative_abundance').fit_transform(self.df)
        self.assertListEqual(result['sample_id'].tolist(), ['S1', 'S2', 'S3'])

    def test_fit_returns_self(self):
        norm = Normalizer(method='relative_abundance')
        self.assertIs(norm.fit(self.df), norm)

    def test_fit_transform_equals_transform(self):
        norm = Normalizer(method='relative_abundance')
        r1 = norm.fit_transform(self.df)
        r2 = norm.transform(self.df)
        pd.testing.assert_frame_equal(r1, r2)


# ── Normalizer – CLR ──────────────────────────────────────────────────────────

class TestNormalizerCLR(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        n = 20
        data = rng.random((n, 5)) + 0.01   # strictly positive
        self.df = pd.DataFrame(data, columns=[f'taxon_{i}' for i in range(5)])
        self.df.insert(0, 'sample_id', [f'S{i:02d}' for i in range(n)])

    def _feature_cols(self, df):
        return [c for c in df.columns if c != 'sample_id']

    def test_row_means_are_zero(self):
        result = Normalizer(method='clr').fit_transform(self.df)
        row_means = result[self._feature_cols(result)].mean(axis=1)
        np.testing.assert_array_almost_equal(
            row_means.values, np.zeros(len(self.df)), decimal=10)

    def test_no_nan_or_inf(self):
        result = Normalizer(method='clr').fit_transform(self.df)
        self.assertFalse(result.isnull().any().any())
        numeric = result[self._feature_cols(result)].values
        self.assertFalse(np.isinf(numeric).any())

    def test_pseudocount_prevents_log_zero(self):
        """Introducing a zero must not produce -inf in the CLR output."""
        df_zeros = self.df.copy()
        df_zeros.iloc[0, 1] = 0.0
        result = Normalizer(method='clr', prevent_zero=1e-6).fit_transform(df_zeros)
        numeric = result[self._feature_cols(result)].values
        self.assertFalse(np.isinf(numeric).any())

    def test_sample_id_preserved(self):
        result = Normalizer(method='clr').fit_transform(self.df)
        self.assertIn('sample_id', result.columns)

    def test_different_pseudocounts_give_different_results(self):
        r1 = Normalizer(method='clr', prevent_zero=1e-9).fit_transform(self.df)
        r2 = Normalizer(method='clr', prevent_zero=1.0).fit_transform(self.df)
        fc = self._feature_cols(r1)
        self.assertFalse(np.allclose(r1[fc].values, r2[fc].values))


class TestNormalizerUnsupportedMethod(unittest.TestCase):
    def test_raises_value_error(self):
        with self.assertRaises(ValueError):
            Normalizer(method='unknown').transform(
                pd.DataFrame({'sample_id': ['S1'], 'a': [1.0]}))


# ── Full Pipeline integration ─────────────────────────────────────────────────

class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.train = _df_with_dups(n=40, seed=10)
        self.val   = _df_with_dups(n=10, seed=20)
        self.test  = _df_with_dups(n=10, seed=30)

    def _pipe(self, merge='pca', norm='clr'):
        return Pipeline([
            ('merger',     TaxonomyMerger(method=merge)),
            ('normalizer', Normalizer(method=norm)),
        ])

    def test_pca_clr_runs_end_to_end(self):
        pipe = self._pipe('pca', 'clr')
        pipe.fit(self.train)
        train_out = pipe.transform(self.train)
        val_out   = pipe.transform(self.val)
        test_out  = pipe.transform(self.test)
        self.assertEqual(train_out.shape[0], 40)
        self.assertEqual(val_out.shape[0],   10)
        self.assertEqual(test_out.shape[0],  10)

    def test_sum_clr_runs_end_to_end(self):
        pipe = self._pipe('sum', 'clr')
        out = pipe.fit_transform(self.train)
        self.assertEqual(out.shape[0], 40)

    def test_pca_relative_abundance_runs(self):
        pipe = self._pipe('pca', 'relative_abundance')
        out = pipe.fit_transform(self.train)
        self.assertEqual(out.shape[0], 40)

    def test_val_test_columns_match_train(self):
        """Val and test output must have the exact same column layout as train."""
        pipe = self._pipe('pca', 'clr')
        pipe.fit(self.train)
        train_out = pipe.transform(self.train)
        val_out   = pipe.transform(self.val)
        test_out  = pipe.transform(self.test)
        self.assertListEqual(train_out.columns.tolist(), val_out.columns.tolist())
        self.assertListEqual(train_out.columns.tolist(), test_out.columns.tolist())

    def test_output_has_no_duplicate_columns(self):
        result = self._pipe('pca', 'clr').fit_transform(self.train)
        self.assertFalse(result.columns.duplicated().any())

    def test_clr_output_row_means_zero(self):
        result = self._pipe('sum', 'clr').fit_transform(self.train)
        fc = [c for c in result.columns if c != 'sample_id']
        np.testing.assert_array_almost_equal(
            result[fc].mean(axis=1).values, np.zeros(40), decimal=10)

    def test_relative_abundance_rows_sum_to_one(self):
        result = self._pipe('sum', 'relative_abundance').fit_transform(self.train)
        fc = [c for c in result.columns if c != 'sample_id']
        np.testing.assert_array_almost_equal(
            result[fc].sum(axis=1).values, np.ones(40))

    def test_pipeline_fit_only_on_train(self):
        """
        PCA min_val is set during fit(train).
        It must not change when transform(val) is subsequently called.
        """
        pipe = self._pipe('pca', 'clr')
        pipe.fit(self.train)
        min_after_fit = pipe.named_steps['merger'].pca_models['taxon_A']['min_val']

        pipe.transform(self.val)
        min_after_transform = pipe.named_steps['merger'].pca_models['taxon_A']['min_val']

        self.assertEqual(min_after_fit, min_after_transform,
            "Pipeline must not refit on validation data")


# ── Data splitting (preprocess logic) ────────────────────────────────────────

class TestDataSplit(unittest.TestCase):
    """
    Tests for the stratified 64 / 16 / 20 train–val–test split
    as implemented in preprocess.py.
    """

    def setUp(self):
        df = _df_labeled(n=100, seed=42)
        X  = df.drop(columns=['study_condition'])
        y  = df['study_condition']

        X_tmp, self.X_test, y_tmp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.2, random_state=42, stratify=y_tmp)

        self.n = len(df)

    def test_test_set_is_20_percent(self):
        self.assertEqual(len(self.X_test), round(self.n * 0.20))

    def test_val_set_is_approx_16_percent(self):
        ratio = len(self.X_val) / self.n
        self.assertAlmostEqual(ratio, 0.16, delta=0.02)

    def test_train_set_is_approx_64_percent(self):
        ratio = len(self.X_train) / self.n
        self.assertAlmostEqual(ratio, 0.64, delta=0.02)

    def test_all_samples_accounted_for(self):
        total = len(self.X_train) + len(self.X_val) + len(self.X_test)
        self.assertEqual(total, self.n)

    def test_no_overlap_train_val(self):
        ids = set(self.X_train['sample_id']) & set(self.X_val['sample_id'])
        self.assertEqual(len(ids), 0, "Train and val must not share samples")

    def test_no_overlap_train_test(self):
        ids = set(self.X_train['sample_id']) & set(self.X_test['sample_id'])
        self.assertEqual(len(ids), 0, "Train and test must not share samples")

    def test_no_overlap_val_test(self):
        ids = set(self.X_val['sample_id']) & set(self.X_test['sample_id'])
        self.assertEqual(len(ids), 0, "Val and test must not share samples")

    def test_stratified_class_balance_train(self):
        full_ratio = (_df_labeled()['study_condition'] == 'CRC').mean()
        train_ratio = (self.y_train == 'CRC').mean()
        self.assertAlmostEqual(train_ratio, full_ratio, delta=0.05)

    def test_stratified_class_balance_val(self):
        full_ratio = (_df_labeled()['study_condition'] == 'CRC').mean()
        val_ratio = (self.y_val == 'CRC').mean()
        self.assertAlmostEqual(val_ratio, full_ratio, delta=0.05)

    def test_stratified_class_balance_test(self):
        full_ratio = (_df_labeled()['study_condition'] == 'CRC').mean()
        test_ratio = (self.y_test == 'CRC').mean()
        self.assertAlmostEqual(test_ratio, full_ratio, delta=0.05)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    unittest.main(verbosity=2)
