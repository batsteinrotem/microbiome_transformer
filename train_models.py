import os
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: render straight to files
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve,
)
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.ERROR)

BASE_DIR = "./"
N_TRIALS = 50 
RANDOM_STATE = 42

# Load data, map crc to 1 and healthy to 0
def load_splits():
    train = pd.read_csv(os.path.join(BASE_DIR, "train_processed.csv"))
    val   = pd.read_csv(os.path.join(BASE_DIR, "val_processed.csv"))
    test  = pd.read_csv(os.path.join(BASE_DIR, "test_processed.csv"))

    drop_cols = ["sample_id", "study_condition"]

    def split_xy(df):
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df["study_condition"].map({'control': 0, 'CRC': 1})

        return X, y

    X_train, y_train = split_xy(train)
    X_val,   y_val   = split_xy(val)
    X_test,  y_test  = split_xy(test)


    return X_train, y_train, X_val, y_val, X_test, y_test

# ── 2. Optuna Objective Factory 
def get_model(trial, model_name):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-4, 10.0, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            solver="liblinear",
            random_state=RANDOM_STATE,
            max_iter=1000
        )

    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            random_state=RANDOM_STATE
        )

    elif model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 4),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True), 
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            scale_pos_weight=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )

    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            num_leaves=trial.suggest_int("num_leaves", 7, 20), # Removed max_depth conflict
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 2, 20), 
            feature_fraction=trial.suggest_float("feature_fraction", 0.1, 1.0),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            random_state=RANDOM_STATE,
            verbose=-1
        )
    
# ── 3. Helper for Evaluation ───────────────────────────────────────────────
def evaluate_split(model, X, y):
    preds_proba = model.predict_proba(X)[:, 1]
    preds_bin = model.predict(X)
    return {
        "auc": roc_auc_score(y, preds_proba),
        "f1": f1_score(y, preds_bin),
        "acc": accuracy_score(y, preds_bin),
        "cm": confusion_matrix(y, preds_bin),
        # Retained for the ROC curve export.
        "proba": preds_proba,
        "y_true": np.asarray(y),
    }

# ── 4. Training & Evaluation Pipeline ──────────────────────────────────────
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    models_to_test = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    results = {}

    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = np.concatenate([y_train, y_val])

    for model_name in models_to_test:
        print(f"Training: {model_name.upper()}")

        # Clean Optuna Trial (No internal early stopping leak)
        def objective(trial):
            model = get_model(trial, model_name)
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)

        best_params = study.best_params
        
        # --- Evaluate Train and Val ---
        eval_model = get_model(optuna.trial.FixedTrial(best_params), model_name)
        eval_model.fit(X_train, y_train)
        
        train_metrics = evaluate_split(eval_model, X_train, y_train)
        val_metrics = evaluate_split(eval_model, X_val, y_val)

        # --- Train FINAL model on Train + Validation combined ---
        final_model = get_model(optuna.trial.FixedTrial(best_params), model_name)
        final_model.fit(X_train_full, y_train_full)
        
        test_metrics = evaluate_split(final_model, X_test, y_test)

        results[model_name] = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics
        }

    return results

# ── 5. Main ─────────────────────────────────────────────────────────────
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    results = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

    records = []
    for m, vals in results.items():
        records.append({
            "Model": m.replace("_", " ").title(),
            "Train_AUC": round(vals["train"]["auc"], 3),
            "Val_AUC": round(vals["val"]["auc"], 3),
            "Test_AUC": round(vals["test"]["auc"], 3),
            "Train_F1": round(vals["train"]["f1"], 3),
            "Val_F1": round(vals["val"]["f1"], 3),
            "Test_F1": round(vals["test"]["f1"], 3),
            "Train_Acc": round(vals["train"]["acc"], 3),
            "Val_Acc": round(vals["val"]["acc"], 3),
            "Test_Acc": round(vals["test"]["acc"], 3)
        })
        
    df_results = pd.DataFrame(records).sort_values(by="Test_AUC", ascending=False)

    out_csv = os.path.join(BASE_DIR, "results_summary_extended.csv")
    df_results.to_csv(out_csv, index=False)

    # Save confusion-matrix plots: one 2x2 grid (all models) per split.
    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax, (m, vals) in zip(axes.ravel(), results.items()):
            ConfusionMatrixDisplay(
                vals[split]["cm"], display_labels=["control", "CRC"]
            ).plot(ax=ax, colorbar=False)
            ax.set_title(m.replace("_", " ").title())
        fig.suptitle(f"Confusion Matrices - {split} set")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"confusion_matrix_{split}.png"), dpi=150)
        plt.close(fig)
    print(f"Saved confusion-matrix plots to {out_dir}/")

    # Combined ROC curve (test set, all models on one graph).
    plt.figure(figsize=(7, 6))
    for m, vals in results.items():
        t = vals["test"]
        fpr, tpr, _ = roc_curve(t["y_true"], t["proba"])
        plt.plot(fpr, tpr, label=f"{m.replace('_', ' ').title()} (AUC={t['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_test.png"), dpi=150)
    plt.close()
    print(f"Saved ROC curve to {os.path.join(out_dir, 'roc_test.png')}")

    print("\nFinal Extended Leaderboard (Sorted by Test AUC):")
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()