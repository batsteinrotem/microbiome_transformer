import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from taxonomy_merger import TaxonomyMerger
from normalize import Normalizer


def load_data(base_dir, study_name):
    print("Loading data...")
    df_abundances = pd.read_csv(os.path.join(base_dir, f"{study_name}_abundances.csv"))
    df_metadata = pd.read_csv(os.path.join(base_dir, f"{study_name}_metadata.csv"))

    # Extract labels and merge
    labels = df_metadata[['sample_id', 'study_condition']]
    df_full = pd.merge(df_abundances, labels, on='sample_id')

    # Drop samples with no label before any further processing
    before = len(df_full)
    df_full = df_full.dropna(subset=['study_condition'])
    dropped = before - len(df_full)
    if dropped:
        print(f"Dropped {dropped} sample(s) with missing study_condition.")

    return df_full


def split_data(df_full):
    # Separate Features (X) and Target (y) BEFORE splitting
    X = df_full.drop(columns=['study_condition'])
    y = df_full['study_condition']

    print("Splitting data into Train, Val, Test")
    # First split: Train+Val vs Test using stratify to balance CRC & Healthy
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Second split: Train vs Val using stratify
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(merger_method="sum", normalizer_method="clr"):
    # Pre-process using scikit pipeline
    pipeline = Pipeline([
        ('merger', TaxonomyMerger(method=merger_method)),
        ('normalizer', Normalizer(method=normalizer_method))
    ])
    return pipeline


def export_data_csv(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test):
    print("Exporting files")
    # Safely reattach labels using DataFrame indices to ensure alignment
    train_export = X_train_processed.copy()
    train_export.insert(1, 'study_condition', y_train)

    val_export = X_val_processed.copy()
    val_export.insert(1, 'study_condition', y_val)

    test_export = X_test_processed.copy()
    test_export.insert(1, 'study_condition', y_test)

    train_export.to_csv("train_processed.csv", index=False)
    val_export.to_csv("val_processed.csv", index=False)
    test_export.to_csv("test_processed.csv", index=False)

    print("Preprocessing complete")


def preprocess(base_dir, study_name, merger_method="sum", normalizer_method="clr", export=False):

    df_full = load_data(base_dir, study_name)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_full)

    pipeline = build_pipeline(merger_method, normalizer_method)

    # Fit ONLY on training data, then transform each split
    X_train_processed = pipeline.fit_transform(X_train)
    X_val_processed   = pipeline.transform(X_val)
    X_test_processed  = pipeline.transform(X_test)

    if export:
        export_data_csv(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test)

    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test


if __name__ == "__main__":
    BASE_DIR = r"c:\Users\batst\OneDrive\Desktop\Microbiome Data\CRC_Healthy_Merged"
    STUDY_NAME = "VogtmannE_2016"
    preprocess(BASE_DIR, STUDY_NAME, export=True)
