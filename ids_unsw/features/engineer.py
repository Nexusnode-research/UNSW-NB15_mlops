# ids_unsw/features/engineer.py
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from ids_unsw.features.preprocessor import CategoricalPreprocessor

CAT_COLS = ["proto", "service", "state", "attack_cat"]

def parse_args():
    p = argparse.ArgumentParser(description="Feature engineering (dedup + categorical encoding) per VTT.")
    p.add_argument("--train-input", required=True, help="Path to UNSW_NB15_training-set.parquet")
    p.add_argument("--test-input",  required=False, help="Path to UNSW_NB15_testing-set.parquet")
    p.add_argument("--train-output", required=True, help="Path to save UNSW_NB15_train_clean.parquet")
    p.add_argument("--test-output",  required=False, help="Path to save UNSW_NB15_test_clean.parquet (if --test-input given)")
    p.add_argument("--preprocessor-out", required=True, help="Where to save the fitted categorical preprocessor (.pkl)")
    return p.parse_args()

def clean_service(df: pd.DataFrame) -> pd.DataFrame:
    if "service" in df.columns:
        df = df.copy()
        df["service"] = df["service"].astype(str).replace("-", "unknown")
    return df



def main():
    args = parse_args()

    train_in  = Path(args.train_input)
    test_in   = Path(args.test_input) if args.test_input else None
    train_out = Path(args.train_output)
    test_out  = Path(args.test_output) if args.test_output else None
    prep_out  = Path(args.preprocessor_out)

    for p in [train_in] + ([test_in] if test_in else []):
        if p and not p.exists():
            raise FileNotFoundError(f"Input parquet not found: {p}")

    train_out.parent.mkdir(parents=True, exist_ok=True)
    if test_out: test_out.parent.mkdir(parents=True, exist_ok=True)
    prep_out.parent.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"[FE] Loading TRAIN: {train_in}")
    train_df = pd.read_parquet(train_in)
    print(f"[FE] Original Train shape: {train_df.shape}")

    test_df = None
    if test_in:
        print(f"[FE] Loading TEST:  {test_in}")
        test_df = pd.read_parquet(test_in)
        print(f"[FE] Original Test shape:  {test_df.shape}")

    # Deduplicate train (as in your cell)
    before = train_df.shape
    train_df = train_df.drop_duplicates(ignore_index=True)
    print(f"[FE] Train shape after removing duplicates: {before} -> {train_df.shape}")

    # Clean 'service' then ensure cat cols exist
    train_df = clean_service(train_df)
    if test_df is not None:
        test_df = clean_service(test_df)

    # Fit encoder on TRAIN cat-cols, transform both
    missing = [c for c in CAT_COLS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing categorical columns in train: {missing}")

    pre = CategoricalPreprocessor(CAT_COLS).fit(train_df)
    train_df = pre.transform(train_df)
    if test_df is not None:
        # If test missing any cat col, raise early
        miss_test = [c for c in CAT_COLS if c not in test_df.columns]
        if miss_test:
            raise ValueError(f"Missing categorical columns in test: {miss_test}")
        test_df = pre.transform(test_df)

    # Save outputs
    print(f"[FE] Saving TRAIN CLEAN → {train_out}")
    train_df.to_parquet(train_out, index=False)
    if test_df is not None and test_out is not None:
        print(f"[FE] Saving TEST CLEAN  → {test_out}")
        test_df.to_parquet(test_out, index=False)

    # Save fitted preprocessor
    with open(prep_out, "wb") as f:
        pickle.dump(pre, f)
    print(f"[FE] Preprocessor saved → {prep_out}")

    print("[FE] DONE.")

if __name__ == "__main__":
    main()
