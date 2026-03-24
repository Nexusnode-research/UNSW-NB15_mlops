from __future__ import annotations
import argparse, os, json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

import mlflow
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

TARGET = "label"
EXPERIMENT_NAME_DEFAULT = "unsw-nb15"

def parse_args():
    p = argparse.ArgumentParser(description="04 Experimentation — Cells 1–5 (setup/load/split/scale + Torch LogReg).")
    p.add_argument("--train-input", required=True, help="Path to UNSW_NB15_train_clean.parquet")
    p.add_argument("--test-input",  required=True, help="Path to UNSW_NB15_test_clean.parquet")
    p.add_argument("--models-dir",  default="notebooks/ids_unsw/models", help="Where to save scaler.pkl etc.")
    p.add_argument("--mlflow-uri",  default="http://host.docker.internal:5000", help="MLflow tracking URI")
    p.add_argument("--mlflow-exp",  default=EXPERIMENT_NAME_DEFAULT, help="MLflow experiment name")
    p.add_argument("--batch-size",  type=int, default=8192)
    p.add_argument("--val-size",    type=float, default=0.15)
    p.add_argument("--seed",        type=int, default=42)
    # Torch logistic regression (Cell 5)
    p.add_argument("--train-logreg", action="store_true", help="Train Torch logistic regression and log to MLflow.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr",     type=float, default=1e-3)
    # XGBoost (Cell 7)
    p.add_argument("--train-xgb", action="store_true", help="Train XGBoost (Cell 7) and log to MLflow.")
    p.add_argument("--xgb-rounds", type=int, default=400)
    p.add_argument("--xgb-early",  type=int, default=50)
    # MLP (Cell 9)
    p.add_argument("--train-mlp", action="store_true", help="Train MLP (Cell 9) and log to MLflow.")
    p.add_argument("--mlp-epochs", type=int, default=10)
    p.add_argument("--mlp-lr",     type=float, default=1e-3)
    # RandomForest (Cell 8)
    p.add_argument("--train-rf", action="store_true", help="Train RandomForest (Cell 8) and log to MLflow.")
    p.add_argument("--rf-n-est", type=int, default=100)
    # Comparison (Cell 10)
    p.add_argument("--compare", action="store_true", help="Compare trained models on TEST with best thresholds (Cell 10).")
    # Artifacts (Cell 11)
    p.add_argument("--save-artifacts", action="store_true",
                   help="Save best_xgboost_model.pkl, best_randomforest_model.pkl, and feature_names.json (Cell 11).")
    # Tuning (Cell 9)
    p.add_argument("--tune-xgb", action="store_true", help="RandomizedSearchCV for XGBoost (Cell 9).")
    p.add_argument("--tune-rf",  action="store_true", help="RandomizedSearchCV for RandomForest (Cell 9).")
    p.add_argument("--tune-iters", type=int, default=10)
    p.add_argument("--tune-cv",    type=int, default=3)
    p.add_argument("--tune-n-jobs", type=int, default=1)  # keep 1 for stability
    return p.parse_args()

def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    out["accuracy"]  = float(accuracy_score(y_true, y_pred))
    out["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    return out

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        bs = xb.size(0)
        running += loss.item() * bs
        n += bs
    return running / max(n, 1)

@torch.inference_mode()
def predict_proba_torch(model, loader, device):
    model.eval()
    probs = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        probs.append(p)
    return np.concatenate(probs, axis=0)

def best_threshold(y_true, y_prob):
    """Grid-search threshold in [0.05, 0.95] to maximize F1."""
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        f1 = f1_score(y_true, (np.asarray(y_prob) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def main():
    args = parse_args()

    # ---- MLflow (Cell 1) ----
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_exp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    # ---- Load (Cell 2) ----
    train_pq = Path(args.train_input)
    test_pq  = Path(args.test_input)
    assert train_pq.exists(), f"Missing file: {train_pq}"
    assert test_pq.exists(),  f"Missing file: {test_pq}"
    train_df = pd.read_parquet(train_pq)
    test_df  = pd.read_parquet(test_pq)
    print("Loaded:")
    print(f"  train_df: {train_df.shape} -> {train_pq.name}")
    print(f"  test_df : {test_df.shape}  -> {test_pq.name}")

    # ---- Features (Cell 3) ----
    drop_cols = [TARGET, "attack_cat"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    print(f"Using {len(feature_cols)} features. 'attack_cat' excluded.")
    print("-" * 30)

    assert set(feature_cols).issubset(set(test_df.columns)), "Test set missing some feature columns."

    non_numeric = [c for c in feature_cols if not is_numeric_dtype(train_df[c])]
    if non_numeric:
        print("Converting non-numeric columns to categorical codes:", non_numeric)
        for c in non_numeric:
            cats = pd.Categorical(train_df[c]).categories
            train_df[c] = pd.Categorical(train_df[c], categories=cats).codes
            test_df[c]  = pd.Categorical(test_df[c],  categories=cats).codes

    X = train_df[feature_cols].astype("float32").values
    y = train_df[TARGET].astype("float32").values
    X_test = test_df[feature_cols].astype("float32").values
    y_test = test_df[TARGET].astype("float32").values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=y, shuffle=True
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print("Features scaled with StandardScaler.")
    print("-" * 30)

    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "scaler.pkl"
    import pickle
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {scaler_path}")

    # Tensors & Loaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    BATCH_SIZE = int(args.batch_size)
    pin = (device.type == "cuda")
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True,  pin_memory=pin, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, num_workers=0)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin, num_workers=0)

    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight_val = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=device)

    print(f"device: {device}")
    print(f"feature_cols: {len(feature_cols)}")
    print(f"train / val / test shapes: {X_train.shape} / {X_val.shape} / {X_test.shape}")
    print(f"y_train pos rate: {pos/len(y_train):.4f} -> pos_weight={float(pos_weight_val):.3f}")
    print("[TRAIN] Step 1 complete (setup/load/split/scale/dataloaders).")

    best_estimators = {}  # holds tuned estimators if we run --tune-*

    # =========================
    # Cell 5 — Torch Logistic Regression (GPU) with MLflow
    # =========================
    if args.train_logreg:
        n_features = X_train.shape[1]
        model = nn.Sequential(nn.Linear(n_features, 1)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        # We'll compute val metrics each epoch using y_val_t available here
        y_val_true = y_val_t.cpu().numpy().ravel()

        with mlflow.start_run(run_name="logreg_torch_gpu") as run:
            mlflow.log_param("model", "LogisticRegression_Torch")
            mlflow.log_param("epochs", int(args.epochs))
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("lr", float(args.lr))
            mlflow.log_param("pos_weight", float(pos_weight_val))
            mlflow.log_param("n_features", int(n_features))
            mlflow.log_param("device", str(device))

            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_prob = predict_proba_torch(model, val_loader, device)
                val_metrics = compute_metrics(y_val_true, val_prob)
                mlflow.log_metric("train_loss", float(train_loss), step=epoch)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", float(v), step=epoch)
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Val AUC: {val_metrics['roc_auc']:.4f}"
                )

            # Last epoch's val_metrics: full dict for papers (precision/recall at τ=0.5)
            print("VAL metrics (final epoch, τ=0.5):", val_metrics)

            exp = mlflow.get_experiment_by_name(args.mlflow_exp)
            if exp is not None:
                print(f"⯑ View run logreg_torch_gpu at: {args.mlflow_uri}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}")
            print("⯑ training complete, model still in memory")

    # =========================
    # Cell 7 — XGBoost (GPU/CPU) with MLflow
    # =========================
    if args.train_xgb:
        try:
            import xgboost as xgb
        except ModuleNotFoundError:
            raise SystemExit(
                "xgboost is not installed in this env. Inside your container run:\n"
                "  pip install xgboost\n"
            )

        use_gpu  = (device.type == "cuda")
        tree_meth = "gpu_hist" if use_gpu else "hist"
        predictor = "gpu_predictor" if use_gpu else "auto"

        params = {
            "max_depth": 6,
            "eta": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "min_child_weight": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": tree_meth,
            "predictor": predictor,
            "scale_pos_weight": float(pos_weight_val.cpu().numpy()),
            "seed": int(args.seed),
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)

        with mlflow.start_run(run_name="xgboost_gpu") as run:
            mlflow.log_param("model", "XGBoost")
            mlflow.log_param("use_gpu", use_gpu)
            mlflow.log_params(params)

            evals = [(dtrain, "train"), (dval, "val")]
            xgb_model = xgb.train(
                params,
                dtrain,
                num_boost_round=int(args.xgb_rounds),
                evals=evals,
                early_stopping_rounds=int(args.xgb_early),
                verbose_eval=False,
            )

            # Validation metrics
            val_prob = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration + 1))
            val_metrics = compute_metrics(y_val, val_prob)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v))

            print(f"⯑ XGBoost trained. use_gpu={use_gpu} | best_iteration={xgb_model.best_iteration}")
            print("VAL metrics:", val_metrics)

            exp = mlflow.get_experiment_by_name(args.mlflow_exp)
            if exp is not None:
                print(f"🏃 View run xgboost_gpu at: {args.mlflow_uri}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}")
                print(f"🧪 View experiment at: {args.mlflow_uri}/#/experiments/{exp.experiment_id}")

    # =========================
    # Cell 9 — MLP (Torch) with MLflow
    # =========================
    if args.train_mlp:
        n_features = X_train.shape[1]
        mlp = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        optimizer = optim.AdamW(mlp.parameters(), lr=float(args.mlp_lr))

        y_val_true  = y_val_t.cpu().numpy().ravel()
        y_test_true = y_test_t.cpu().numpy().ravel()

        with mlflow.start_run(run_name="mlp_torch_gpu") as run:
            mlflow.log_params({
                "model": "MLP_Torch",
                "device": str(device),
                "epochs": int(args.mlp_epochs),
                "batch_size": int(BATCH_SIZE),
                "lr": float(args.mlp_lr),
                "pos_weight": float(pos_weight_val),
                "n_features": int(n_features),
                "hidden1": 128,
                "hidden2": 64,
                "dropout": 0.2,
            })

            # Train for N epochs, logging VAL metrics each epoch
            for epoch in range(1, int(args.mlp_epochs) + 1):
                train_loss = train_one_epoch(mlp, train_loader, criterion, optimizer, device)
                val_prob   = predict_proba_torch(mlp, val_loader, device)
                val_metrics = compute_metrics(y_val_true, val_prob)
                mlflow.log_metric("train_loss", float(train_loss), step=epoch)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", float(v), step=epoch)
                print(
                    f"Epoch {epoch}/{args.mlp_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Val AUC: {val_metrics['roc_auc']:.4f}"
                )

            # Final TEST evaluation + logging
            test_prob   = predict_proba_torch(mlp, test_loader, device)
            test_metrics = compute_metrics(y_test_true, test_prob)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", float(v))

            exp = mlflow.get_experiment_by_name(args.mlflow_exp)
            if exp is not None:
                print(f"🏃 View run mlp_torch_gpu at: {args.mlflow_uri}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}")
                print(f"🧪 View experiment at: {args.mlflow_uri}/#/experiments/{exp.experiment_id}")

    # =========================
    # Cell 8 — RandomForest (CPU) with MLflow
    # =========================
    if args.train_rf:
        from sklearn.ensemble import RandomForestClassifier

        rf_model = RandomForestClassifier(
            n_estimators=int(args.rf_n_est),
            random_state=int(args.seed),
            n_jobs=-1,
            class_weight="balanced",
        )

        print("Training RandomForestClassifier...")
        with mlflow.start_run(run_name="random_forest_cpu") as run:
            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_param("n_estimators", int(args.rf_n_est))
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("random_state", int(args.seed))

            rf_model.fit(X_train, y_train)
            print("⯑ RandomForest trained.")

            val_prob = rf_model.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, val_prob)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v))

            print("\nVAL metrics:", val_metrics)

            exp = mlflow.get_experiment_by_name(args.mlflow_exp)
            if exp is not None:
                print(f"🏃 View run random_forest_cpu at: {args.mlflow_uri}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}")
                print(f"🧪 View experiment at: {args.mlflow_uri}/#/experiments/{exp.experiment_id}")

    # =========================
    # Cell 9 — Stable Tuning with RandomizedSearchCV
    # =========================
    if args.tune_rf:
        from sklearn.ensemble import RandomForestClassifier
        with mlflow.start_run(run_name="random_tuning_RandomForest") as parent_run:
            rf = RandomForestClassifier(random_state=int(args.seed), class_weight="balanced")
            rf_params = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            print("\n--- Tuning RandomForest with RandomizedSearchCV ---")
            rs = RandomizedSearchCV(
                estimator=rf,
                param_distributions=rf_params,
                n_iter=int(args.tune_iters),
                cv=int(args.tune_cv),
                scoring="f1",
                verbose=1,
                random_state=int(args.seed),
                n_jobs=int(args.tune_n_jobs),
            )
            rs.fit(X_train, y_train)
            print(f"⯑ Best parameters for RandomForest: {rs.best_params_}")
            print(f"⯑ Best CV F1: {rs.best_score_:.4f}")
            mlflow.log_metric("best_cv_f1_score", float(rs.best_score_))
            mlflow.log_params({f"best_{k}": v for k, v in rs.best_params_.items()})
            best_estimators["RandomForest"] = rs.best_estimator_

    if args.tune_xgb:
        try:
            import xgboost as xgb
        except ModuleNotFoundError:
            raise SystemExit("Need xgboost installed. Try: pip install xgboost")

        with mlflow.start_run(run_name="random_tuning_XGBoost") as parent_run:
            xgb_clf = xgb.XGBClassifier(
                random_state=int(args.seed),
                eval_metric="auc",
                tree_method="gpu_hist" if (torch.cuda.is_available()) else "hist",
                predictor="gpu_predictor" if (torch.cuda.is_available()) else "auto",
                scale_pos_weight=float(pos_weight_val.cpu().numpy()),
                use_label_encoder=False,
            )
            xgb_params = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [5, 7, 10],
                "subsample": [0.7, 0.8, 1.0],
            }
            print("\n--- Tuning XGBoost with RandomizedSearchCV ---")
            rs = RandomizedSearchCV(
                estimator=xgb_clf,
                param_distributions=xgb_params,
                n_iter=int(args.tune_iters),
                cv=int(args.tune_cv),
                scoring="f1",
                verbose=1,
                random_state=int(args.seed),
                n_jobs=int(args.tune_n_jobs),
            )
            rs.fit(X_train, y_train)
            print(f"⯑ Best parameters for XGBoost: {rs.best_params_}")
            print(f"⯑ Best CV F1: {rs.best_score_:.4f}")
            mlflow.log_metric("best_cv_f1_score", float(rs.best_score_))
            mlflow.log_params({f"best_{k}": v for k, v in rs.best_params_.items()})
            best_estimators["XGBoost"] = rs.best_estimator_

    # =========================
    # Cell 10 — Compare available models on TEST with per-model best thresholds
    # =========================
    if args.compare:
        rows = []
        cols = ["model", "val_best_thr", "roc_auc", "accuracy", "f1", "precision", "recall"]

        # Ground-truth numpy arrays
        y_val_true  = y_val_t.cpu().numpy().ravel()
        y_test_true = y_test_t.cpu().numpy().ravel()

        # ---- Logistic (Torch) if present ----
        if "model" in locals():
            val_prob_log = predict_proba_torch(model, val_loader, device)
            t_log, _ = best_threshold(y_val_true, val_prob_log)
            test_prob_log = predict_proba_torch(model, test_loader, device)
            m_log = compute_metrics(y_test_true, test_prob_log, threshold=t_log)
            rows.append({"model": "LogReg_Torch", "val_best_thr": t_log, **m_log})
        else:
            print("↪ Skipping logistic: variable `model` not found (run with --train-logreg).")

        # ---- XGBoost if present ----
        if "xgb_model" in locals():
            import xgboost as xgb
            dval  = xgb.DMatrix(X_val,  label=y_val)
            dtest = xgb.DMatrix(X_test, label=y_test)
            val_prob_xgb = xgb_model.predict(dval,  iteration_range=(0, xgb_model.best_iteration + 1))
            t_xgb, _ = best_threshold(y_val, val_prob_xgb)
            test_prob_xgb = xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1))
            m_xgb = compute_metrics(y_test, test_prob_xgb, threshold=t_xgb)
            rows.append({"model": "XGBoost", "val_best_thr": t_xgb, **m_xgb})
        else:
            print("↪ Skipping XGBoost: variable `xgb_model` not found (run with --train-xgb).")

        # ---- MLP (Torch) if present ----
        if "mlp" in locals():
            val_prob_mlp = predict_proba_torch(mlp, val_loader, device)
            t_mlp, _ = best_threshold(y_val_true, val_prob_mlp)
            test_prob_mlp = predict_proba_torch(mlp, test_loader, device)
            m_mlp = compute_metrics(y_test_true, test_prob_mlp, threshold=t_mlp)
            rows.append({"model": "MLP_Torch", "val_best_thr": t_mlp, **m_mlp})
        else:
            print("↪ Skipping MLP: variable `mlp` not found (run with --train-mlp).")

        # ---- RandomForest if present ----
        if "rf_model" in locals():
            val_prob_rf = rf_model.predict_proba(X_val)[:, 1]
            t_rf, _ = best_threshold(y_val, val_prob_rf)
            test_prob_rf = rf_model.predict_proba(X_test)[:, 1]
            m_rf = compute_metrics(y_test, test_prob_rf, threshold=t_rf)
            rows.append({"model": "RandomForest", "val_best_thr": t_rf, **m_rf})
        else:
            print("↪ Skipping RandomForest: variable `rf_model` not found (run with --train-rf).")
            
        # ---- Tuned RandomForest if present ----
        if "RandomForest" in best_estimators:
            tuned_rf = best_estimators["RandomForest"]
            val_prob_rf = tuned_rf.predict_proba(X_val)[:, 1]
            t_rf, _ = best_threshold(y_val, val_prob_rf)
            test_prob_rf = tuned_rf.predict_proba(X_test)[:, 1]
            m_rf = compute_metrics(y_test, test_prob_rf, threshold=t_rf)
            rows.append({"model": "RandomForest_Tuned", "val_best_thr": t_rf, **m_rf})

        # ---- Tuned XGBoost if present ----
        if "XGBoost" in best_estimators:
            tuned_xgb = best_estimators["XGBoost"]
            val_prob_xgb = tuned_xgb.predict_proba(X_val)[:, 1]
            t_xgb, _ = best_threshold(y_val, val_prob_xgb)
            test_prob_xgb = tuned_xgb.predict_proba(X_test)[:, 1]
            m_xgb = compute_metrics(y_test, test_prob_xgb, threshold=t_xgb)
            rows.append({"model": "XGBoost_Tuned", "val_best_thr": t_xgb, **m_xgb})

        if rows:
            df = pd.DataFrame(rows)[cols].sort_values(by=["roc_auc", "f1"], ascending=False)
            print("\n=== TEST Comparison ===")
            print(df.to_string(index=False))
        else:
            print("No models to compare. Train some in this run and re-run with --compare.")

    # =========================
    # Cell 11 — Create Model Artifacts
    # =========================
    if args.save_artifacts:
        import pickle, json
        models_dir = Path(args.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        # 1) Save XGBoost (prefer tuned)
        xgb_out = models_dir / "best_xgboost_model.pkl"
        xgb_to_save = best_estimators.get("XGBoost") if "XGBoost" in best_estimators else locals().get("xgb_model")
        if xgb_to_save is not None:
            with open(xgb_out, "wb") as f:
                pickle.dump(xgb_to_save, f)
            print(f"⯑ Best XGBoost model saved to: {xgb_out}")
        else:
            print("↪ Could not save best_xgboost_model.pkl: no XGBoost model found.")
        
        # 2) Save RandomForest (prefer tuned)
        rf_out = models_dir / "best_randomforest_model.pkl"
        rf_to_save = best_estimators.get("RandomForest") if "RandomForest" in best_estimators else locals().get("rf_model")
        if rf_to_save is not None:
            with open(rf_out, "wb") as f:
                pickle.dump(rf_to_save, f)
            print(f"⯑ Best Random Forest model saved to: {rf_out}")
        else:
            print("↪ Could not save best_randomforest_model.pkl: no RF model found.")

        # 3) Save feature names
        feats_out = models_dir / "feature_names.json"
        with open(feats_out, "w") as f:
            json.dump({"features": list(feature_cols)}, f, indent=2)
        print(f"⯑ Feature names saved to: {feats_out}")

        print(f"Number of features the model was trained on: {len(feature_cols)}")


if __name__ == "__main__":
    main()

