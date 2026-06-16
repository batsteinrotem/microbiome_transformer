"""End-to-end entry point: preprocess raw abundances, then train + evaluate models.

Step 1 (preprocess.py): load the raw study, split, run the TaxonomyMerger -> CLR
pipeline (fit on train only), and return the processed splits in memory.
Step 2 (train_models.py): consume those DataFrames directly, tune + train the four
models, and write all result artifacts (metrics, confusion matrices, ROC, models,
raw predictions). No intermediate CSVs are needed.
"""
import preprocess as pp
import train_models as tm

# Raw-data location for the preprocessing step.
BASE_DIR = r"c:\Users\batst\OneDrive\Desktop\Microbiome Data\CRC_Healthy_Merged"
STUDY_NAME = "VogtmannE_2016"
MERGER_METHOD = "sum"
NORMALIZER_METHOD = "clr"


def main():
    print("=" * 60)
    print("STEP 1 / 2  -  PREPROCESSING")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = pp.preprocess(
        base_dir=BASE_DIR,
        study_name=STUDY_NAME,
        merger_method=MERGER_METHOD,
        normalizer_method=NORMALIZER_METHOD,
        export=False,            # hand the DataFrames straight to training, no CSVs
    )

    print("\n" + "=" * 60)
    print("STEP 2 / 2  -  TRAINING & EVALUATION")
    print("=" * 60)
    tm.run_from_dataframes(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()
