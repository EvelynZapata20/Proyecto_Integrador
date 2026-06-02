"""
Modelado supervisado unificado para clasificación binaria desbalanceada.

Entrada esperada:
    - Dataset final con variables definitivas SIN codificar.
    - Debe contener TARGET y, opcionalmente, SK_ID_CURR.

Qué hace:
    1. Lee csv/parquet/feather.
    2. Excluye TARGET e ID de predictores.
    3. Separa variables en:
        - numéricas continuas
        - numéricas binarias
        - categóricas binarias
        - categóricas nominales
    4. Codificación:
        - Categóricas nominales: One-Hot Encoding.
        - Categóricas binarias: OrdinalEncoder 0/1, no One-Hot.
        - Numéricas binarias: se dejan como 0/1.
        - Numéricas continuas: imputación + escalado.
    5. Evalúa modelos base e imbalance learning con validación cruzada estratificada.
    6. Selecciona el mejor modelo usando PR-AUC en validación cruzada.
    7. Ajusta threshold optimizando F2 sobre predicciones out-of-fold.
    8. Entrena el mejor modelo final y evalúa en test holdout.
    9. Guarda métricas, curvas, metadata y pipeline final.

Uso ejemplo:
    python 10_unified_cv_imbalance_training.py \
        --data-path data/train_selected_features.parquet \
        --target TARGET \
        --id-col SK_ID_CURR \
        --output-dir data/model_outputs/unified_cv_imbalance \
        --experiment-set full

Dependencias recomendadas:
    pip install pandas numpy scikit-learn imbalanced-learn xgboost pyarrow joblib matplotlib
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 42


# =============================================================================
# Configuración / utilidades generales
# =============================================================================

@dataclass
class FeatureGroups:
    numeric_continuous_cols: List[str]
    numeric_binary_cols: List[str]
    categorical_binary_cols: List[str]
    categorical_nominal_cols: List[str]


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name)).strip("_")
    return name or "model"


def read_dataset(path: Path) -> pd.DataFrame:
    """Lee csv, parquet o feather."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if suffix in [".feather", ".ft"]:
        return pd.read_feather(path)
    raise ValueError(f"Formato no soportado: {path}. Usa csv, parquet o feather.")


def load_features(
    df_columns: Iterable[str],
    target: str,
    id_col: Optional[str],
    drop_cols: Optional[List[str]] = None,
) -> List[str]:
    """Usa todas las columnas disponibles, excluyendo target, id y columnas solicitadas."""
    excluded = {target}
    if id_col:
        excluded.add(id_col)

    drop_cols = drop_cols or []
    excluded.update(drop_cols)

    features = [c for c in df_columns if c not in excluded]

    print(f"[INFO] Variables disponibles para entrenar: {len(features):,}")
    if drop_cols:
        existing_drop = [c for c in drop_cols if c in df_columns]
        missing_drop = [c for c in drop_cols if c not in df_columns]
        print(f"[INFO] Variables eliminadas por configuración: {len(existing_drop):,}")
        if existing_drop:
            print(f"       {existing_drop}")
        if missing_drop:
            print(f"[AVISO] Variables a eliminar no encontradas: {missing_drop}")

    if not features:
        raise ValueError("No quedó ninguna variable disponible para entrenar.")

    return features


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatibilidad scikit-learn: sparse_output vs sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_ordinal_encoder() -> OrdinalEncoder:
    """Encoder para categóricas binarias. Unknown queda como -1."""
    try:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        )
    except TypeError:
        # Fallback para versiones antiguas.
        return OrdinalEncoder(dtype=np.float32)


def is_numeric_binary(series: pd.Series) -> bool:
    """Detecta variables numéricas binarias tipo 0/1, True/False o dos valores."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_values = pd.unique(non_null)
    if len(unique_values) != 2:
        return False

    # Caso ideal: valores exactamente 0/1 o bool.
    try:
        as_float = pd.Series(unique_values).astype(float)
        return set(as_float.tolist()).issubset({0.0, 1.0})
    except Exception:
        return False


def infer_feature_groups(X: pd.DataFrame) -> FeatureGroups:
    """
    Separa variables para que One-Hot se aplique solo a categóricas no binarias.
    """
    categorical_candidate_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    numeric_candidate_cols = [
        c for c in X.columns
        if c not in categorical_candidate_cols and c not in bool_cols
    ]

    categorical_binary_cols = []
    categorical_nominal_cols = []
    for col in categorical_candidate_cols:
        n_unique = X[col].dropna().nunique()
        if n_unique <= 2:
            categorical_binary_cols.append(col)
        else:
            categorical_nominal_cols.append(col)

    numeric_binary_cols = []
    numeric_continuous_cols = []

    # Bool se trata como binaria numérica.
    numeric_binary_cols.extend(bool_cols)

    for col in numeric_candidate_cols:
        if is_numeric_binary(X[col]):
            numeric_binary_cols.append(col)
        else:
            numeric_continuous_cols.append(col)

    return FeatureGroups(
        numeric_continuous_cols=numeric_continuous_cols,
        numeric_binary_cols=numeric_binary_cols,
        categorical_binary_cols=categorical_binary_cols,
        categorical_nominal_cols=categorical_nominal_cols,
    )


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, FeatureGroups]:
    """
    Preprocesamiento:
      - numéricas continuas: mediana + escalado
      - numéricas binarias: moda, sin One-Hot
      - categóricas binarias: moda + OrdinalEncoder 0/1
      - categóricas nominales: moda + One-Hot
    """
    groups = infer_feature_groups(X)

    transformers = []

    if groups.numeric_continuous_cols:
        numeric_continuous_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num_cont", numeric_continuous_pipe, groups.numeric_continuous_cols))

    if groups.numeric_binary_cols:
        numeric_binary_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("num_bin", numeric_binary_pipe, groups.numeric_binary_cols))

    if groups.categorical_binary_cols:
        categorical_binary_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", make_ordinal_encoder()),
            ]
        )
        transformers.append(("cat_bin", categorical_binary_pipe, groups.categorical_binary_cols))

    if groups.categorical_nominal_cols:
        categorical_nominal_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat_nom", categorical_nominal_pipe, groups.categorical_nominal_cols))

    if not transformers:
        raise ValueError("No se encontraron variables transformables para modelar.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor, groups


# =============================================================================
# Métricas, threshold y gráficas
# =============================================================================

def get_positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Devuelve probabilidad de clase positiva TARGET=1."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise ValueError("El modelo no tiene predict_proba ni decision_function.")


def metrics_at_threshold(y_true: pd.Series | np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f2",
) -> Tuple[float, pd.DataFrame]:
    """Busca threshold entre 0.01 y 0.99 maximizando la métrica dada."""
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows = [metrics_at_threshold(y_true, y_prob, thr) for thr in thresholds]
    table = pd.DataFrame(rows)
    best_idx = table[metric].idxmax()
    return float(table.loc[best_idx, "threshold"]), table


def save_curves(y_true, y_prob, model_name: str, output_dir: Path) -> None:
    """Guarda curva ROC y Precision-Recall."""
    safe_model_name = sanitize_name(model_name)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Azar")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{safe_model_name}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"pr_curve_{safe_model_name}.png", dpi=150)
    plt.close()


def save_comparison_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return

    plot_df = summary.sort_values("oof_pr_auc", ascending=True)

    plt.figure(figsize=(11, max(6, len(plot_df) * 0.45)))
    plt.barh(plot_df["model"], plot_df["oof_pr_auc"])
    plt.xlabel("PR-AUC OOF")
    plt.title("Comparación de modelos por PR-AUC out-of-fold")
    plt.grid(axis="x", alpha=0.3)
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        plt.text(row["oof_pr_auc"] + 0.002, idx, f"{row['oof_pr_auc']:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_oof_pr_auc.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_df = summary.sort_values("oof_roc_auc", ascending=True)
    plt.figure(figsize=(11, max(6, len(plot_df) * 0.45)))
    plt.barh(plot_df["model"], plot_df["oof_roc_auc"])
    plt.xlabel("ROC-AUC OOF")
    plt.title("Comparación de modelos por ROC-AUC out-of-fold")
    plt.grid(axis="x", alpha=0.3)
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        plt.text(row["oof_roc_auc"] + 0.002, idx, f"{row['oof_roc_auc']:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_oof_roc_auc.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Modelos
# =============================================================================

def try_import_imblearn():
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler, TomekLinks
        return {
            "ImbPipeline": ImbPipeline,
            "RandomOverSampler": RandomOverSampler,
            "SMOTE": SMOTE,
            "ADASYN": ADASYN,
            "RandomUnderSampler": RandomUnderSampler,
            "TomekLinks": TomekLinks,
        }
    except ImportError:
        return None


def try_import_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


def make_rf_weighted(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def make_rf_unweighted(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=75,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )


def make_xgb_model(
    XGBClassifier,
    random_state: int = RANDOM_STATE,
    scale_pos_weight: Optional[float] = None,
):
    kwargs = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10,
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = float(scale_pos_weight)
    return XGBClassifier(**kwargs)


def build_models(
    preprocessor: ColumnTransformer,
    y_train_val: pd.Series,
    experiment_set: str = "quick",
) -> Dict[str, object]:
    """
    Construye modelos base + imbalance learning.

    experiment_set:
        - quick: modelos principales y samplers menos costosos.
        - full : agrega SMOTE, ADASYN y TomekLinks para RF y XGBoost.
    """
    models: Dict[str, object] = {}

    models["dummy_prior"] = Pipeline([
        ("preprocess", preprocessor),
        ("model", DummyClassifier(strategy="prior", random_state=RANDOM_STATE)),
    ])

    # Baseline interpretable con peso de clases.
    # NOTA DE RENDIMIENTO:
    #   Antes usaba solver="saga" + max_iter=5000. Con penalización L2 y datos
    #   escalados pero NO centrados (StandardScaler(with_mean=False), obligado por
    #   la matriz dispersa del One-Hot), saga converge lentísimo: un solo fit sobre
    #   ~246k filas no terminaba en varios minutos, y eso x5 folds disparaba el tiempo.
    #   lbfgs es el solver indicado para L2, maneja bien datos no centrados y converge
    #   en ~60 iteraciones. Medido sobre el dataset: ~17 s por fit (vs >250 s con saga),
    #   con PR-AUC y ROC-AUC idénticos hasta la 3a cifra decimal.
    #   tol=1e-3 corta al converger sin afectar las métricas de selección.
    #   IMPORTANTE: si en el futuro se usa penalty="l1" o "elasticnet", volver a saga
    #   (lbfgs no las soporta), pero manteniendo max_iter moderado (~1000) y tol alto.
    models["logistic_balanced"] = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            tol=1e-3,
            random_state=RANDOM_STATE,
        )),
    ])

    # Baseline Random Forest con class_weight.
    models["rf_balanced"] = Pipeline([
        ("preprocess", preprocessor),
        ("model", make_rf_weighted()),
    ])

    XGBClassifier = try_import_xgboost()
    if XGBClassifier is not None:
        scale_pos_weight = y_train_val.value_counts().get(0, 0) / max(y_train_val.value_counts().get(1, 1), 1)
        models["xgb_scale_pos_weight"] = Pipeline([
            ("preprocess", preprocessor),
            ("model", make_xgb_model(XGBClassifier, scale_pos_weight=scale_pos_weight)),
        ])
    else:
        print("[AVISO] xgboost no está instalado. Se omiten modelos XGBoost.")

    imblearn_objs = try_import_imblearn()
    if imblearn_objs is None:
        print("[AVISO] imbalanced-learn no está instalado. Se omiten samplers.")
        return models

    ImbPipeline = imblearn_objs["ImbPipeline"]
    RandomOverSampler = imblearn_objs["RandomOverSampler"]
    SMOTE = imblearn_objs["SMOTE"]
    ADASYN = imblearn_objs["ADASYN"]
    RandomUnderSampler = imblearn_objs["RandomUnderSampler"]
    TomekLinks = imblearn_objs["TomekLinks"]

    ratio_base = 0.25  # minoritaria / mayoritaria = 0.25 => aprox 20% positivos tras sampling.

    # Imbalance learning con Random Forest.
    models["rf_random_over"] = ImbPipeline([
        ("preprocess", preprocessor),
        ("sampler", RandomOverSampler(sampling_strategy=ratio_base, random_state=RANDOM_STATE)),
        ("model", make_rf_unweighted()),
    ])

    models["rf_random_under"] = ImbPipeline([
        ("preprocess", preprocessor),
        ("sampler", RandomUnderSampler(sampling_strategy=ratio_base, random_state=RANDOM_STATE)),
        ("model", make_rf_unweighted()),
    ])

    # Imbalance learning con XGBoost.
    if XGBClassifier is not None:
        models["xgb_random_over"] = ImbPipeline([
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(sampling_strategy=ratio_base, random_state=RANDOM_STATE)),
            ("model", make_xgb_model(XGBClassifier, scale_pos_weight=None)),
        ])

        models["xgb_random_under"] = ImbPipeline([
            ("preprocess", preprocessor),
            ("sampler", RandomUnderSampler(sampling_strategy=ratio_base, random_state=RANDOM_STATE)),
            ("model", make_xgb_model(XGBClassifier, scale_pos_weight=None)),
        ])

    if experiment_set == "full":
        # SMOTE / ADASYN pueden crear valores intermedios en one-hot; se dejan como experimento.
        models["rf_smote"] = ImbPipeline([
            ("preprocess", preprocessor),
            ("sampler", SMOTE(sampling_strategy=ratio_base, random_state=RANDOM_STATE)),
            ("model", make_rf_unweighted()),
        ])

        models["rf_adasyn"] = ImbPipeline([
            ("preprocess", preprocessor),
            ("sampler", ADASYN(sampling_strategy=ratio_base, random_state=RANDOM_STATE)),
            ("model", make_rf_unweighted()),
        ])

        models["rf_tomek_links"] = ImbPipeline([
            ("preprocess", preprocessor),
            ("sampler", TomekLinks()),
            ("model", make_rf_unweighted()),
        ])

        if XGBClassifier is not None:
            models["xgb_smote"] = ImbPipeline([
                ("preprocess", preprocessor),
                ("sampler", SMOTE(sampling_strategy=ratio_base, random_state=RANDOM_STATE)),
                ("model", make_xgb_model(XGBClassifier, scale_pos_weight=None)),
            ])

            models["xgb_adasyn"] = ImbPipeline([
                ("preprocess", preprocessor),
                ("sampler", ADASYN(sampling_strategy=ratio_base, random_state=RANDOM_STATE)),
                ("model", make_xgb_model(XGBClassifier, scale_pos_weight=None)),
            ])

            models["xgb_tomek_links"] = ImbPipeline([
                ("preprocess", preprocessor),
                ("sampler", TomekLinks()),
                ("model", make_xgb_model(XGBClassifier, scale_pos_weight=None)),
            ])

    return models


# =============================================================================
# Validación cruzada, selección y evaluación final
# =============================================================================

def evaluate_cv_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    output_dir: Path,
    threshold_metric: str = "f2",
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray]:
    """Evalúa un modelo con CV, devuelve métricas por fold y resumen OOF."""
    fold_rows = []
    oof_proba = np.zeros(len(y), dtype=np.float32)

    print("\n" + "=" * 90)
    print(f"[INFO] Validación cruzada: {model_name}")
    print("=" * 90)

    y_np = y.to_numpy()

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[INFO] Fold {fold}/{cv.get_n_splits()} - {model_name}")
        fitted_model = clone(model)

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted_model.fit(X_train_fold, y_train_fold)
            if caught:
                for w in caught[:3]:
                    print(f"[AVISO] {model_name} fold {fold}: {w.message}")
                if len(caught) > 3:
                    print(f"[AVISO] {model_name} fold {fold}: {len(caught) - 3} warnings adicionales omitidos.")

        val_prob = get_positive_proba(fitted_model, X_val_fold)
        oof_proba[val_idx] = val_prob.astype(np.float32)

        fold_default = metrics_at_threshold(y_val_fold, val_prob, threshold=0.50)
        fold_default.update({"model": model_name, "fold": fold, "threshold_type": "default_0_50"})
        fold_rows.append(fold_default)

        del fitted_model
        gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)

    best_threshold, threshold_table = find_best_threshold(y_np, oof_proba, metric=threshold_metric)
    threshold_table.insert(0, "model", model_name)
    threshold_table.to_csv(output_dir / f"threshold_search_cv_{sanitize_name(model_name)}.csv", index=False)

    oof_default = metrics_at_threshold(y_np, oof_proba, threshold=0.50)
    oof_best = metrics_at_threshold(y_np, oof_proba, threshold=best_threshold)

    summary = {
        "model": model_name,
        "best_threshold_oof": best_threshold,
        "oof_roc_auc": oof_best["roc_auc"],
        "oof_pr_auc": oof_best["pr_auc"],
        "oof_f2": oof_best["f2"],
        "oof_f1": oof_best["f1"],
        "oof_precision": oof_best["precision"],
        "oof_recall": oof_best["recall"],
        "oof_specificity": oof_best["specificity"],
        "oof_balanced_accuracy": oof_best["balanced_accuracy"],
        "oof_precision_at_0_50": oof_default["precision"],
        "oof_recall_at_0_50": oof_default["recall"],
        "cv_roc_auc_mean": fold_metrics["roc_auc"].mean(),
        "cv_roc_auc_std": fold_metrics["roc_auc"].std(),
        "cv_pr_auc_mean": fold_metrics["pr_auc"].mean(),
        "cv_pr_auc_std": fold_metrics["pr_auc"].std(),
        "cv_f1_mean_at_0_50": fold_metrics["f1"].mean(),
        "cv_recall_mean_at_0_50": fold_metrics["recall"].mean(),
        "cv_precision_mean_at_0_50": fold_metrics["precision"].mean(),
    }

    print("\n[OOF] Métricas con threshold ajustado:")
    print(pd.Series(oof_best).round(4).to_string())
    
    with open(f"metricas_{model_name}.txt", "w", encoding="utf-8") as f:
        f.write("\n[OOF] Métricas con threshold ajustado:\n")
        f.write(pd.Series(oof_best).round(4).to_string())
        f.write("\n")
    
    return fold_metrics, summary, threshold_table, oof_proba


def select_best_model(summary_df: pd.DataFrame) -> pd.Series:
    """Selecciona mejor modelo. Métrica principal: PR-AUC OOF.

    Garantiza que SIEMPRE devuelve una pd.Series válida o lanza una excepción
    explícita. Nunca devuelve None ni un DataFrame vacío, para que el llamador
    pueda usar el resultado sin riesgo de NameError/AttributeError.
    """
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df está vacío: no hay modelos para seleccionar.")

    if "model" not in summary_df.columns:
        raise KeyError("summary_df no tiene columna 'model'.")

    if "oof_pr_auc" not in summary_df.columns:
        raise KeyError("summary_df no tiene columna 'oof_pr_auc' para ordenar.")

    candidates = summary_df[~summary_df["model"].eq("dummy_prior")].copy()
    if candidates.empty:
        # Solo quedó el dummy: lo permitimos como fallback pero avisamos.
        print("[AVISO] El único modelo disponible es 'dummy_prior'. Se selecciona por falta de alternativas.")
        candidates = summary_df.copy()

    candidates = candidates.sort_values(
        ["oof_pr_auc", "oof_roc_auc", "oof_f2"],
        ascending=False,
    )
    return candidates.iloc[0]


def fit_final_and_evaluate(
    best_model_name: str,
    best_model,
    best_threshold: float,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> Tuple[object, pd.DataFrame]:
    """Entrena el mejor modelo en train_val y evalúa en test holdout."""
    print("\n" + "=" * 90)
    print(f"[INFO] Entrenando modelo final: {best_model_name}")
    print("=" * 90)

    final_model = clone(best_model)
    final_model.fit(X_train_val, y_train_val)

    test_prob = get_positive_proba(final_model, X_test)

    rows = []
    for threshold_type, threshold in [
        ("default_0_50", 0.50),
        ("best_oof_threshold", best_threshold),
    ]:
        row = metrics_at_threshold(y_test, test_prob, threshold=threshold)
        row["model"] = best_model_name
        row["split"] = "test"
        row["threshold_type"] = threshold_type
        rows.append(row)

    final_metrics = pd.DataFrame(rows)
    final_metrics.to_csv(output_dir / "final_test_metrics.csv", index=False)

    y_pred = (test_prob >= best_threshold).astype(int)
    report_txt = classification_report(y_test, y_pred, digits=4, zero_division=0)
    with open(output_dir / f"classification_report_final_{sanitize_name(best_model_name)}.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    save_curves(y_test, test_prob, f"final_{best_model_name}", output_dir)

    print("\n[TEST] Métricas finales con threshold OOF ajustado:")
    print(pd.Series(rows[-1]).drop(labels=["model", "split", "threshold_type"]).round(4).to_string())

    return final_model, final_metrics


# =============================================================================
# Orquestación principal
# =============================================================================

def train_and_select(
    data_path: Path,
    target: str,
    id_col: Optional[str],
    output_dir: Path,
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.20,
    n_splits: int = 5,
    experiment_set: str = "quick",
    selection_metric: str = "pr_auc",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset(data_path)
    print(f"[INFO] Dataset leído: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")

    if target not in df.columns:
        raise ValueError(f"No existe la columna target: {target}")

    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)

    if id_col and id_col in df.columns:
        duplicated_ids = int(df[id_col].duplicated().sum())
        print(f"[INFO] Duplicados en {id_col}: {duplicated_ids:,}")

    class_counts = df[target].value_counts().sort_index()
    class_rate = df[target].value_counts(normalize=True).sort_index()
    print("\n[INFO] Distribución del target:")
    for cls in class_counts.index:
        print(f"  Clase {cls}: {class_counts.loc[cls]:,} ({class_rate.loc[cls]:.2%})")

    selected_features = load_features(
        df_columns=df.columns.tolist(),
        target=target,
        id_col=id_col,
        drop_cols=drop_cols,
    )

    X = df[selected_features].copy()
    y = df[target].copy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\n[INFO] Split final:")
    print(f"  Train+CV : {len(X_train_val):,}")
    print(f"  Test     : {len(X_test):,}")

    preprocessor, groups = build_preprocessor(X_train_val)

    print("\n[INFO] Tipos de variables detectadas:")
    print(f"  Numéricas continuas     : {len(groups.numeric_continuous_cols):,}")
    print(f"  Numéricas binarias      : {len(groups.numeric_binary_cols):,}")
    print(f"  Categóricas binarias    : {len(groups.categorical_binary_cols):,}")
    print(f"  Categóricas nominales   : {len(groups.categorical_nominal_cols):,}")
    print("[INFO] Codificación:")
    print("  - Categóricas nominales -> One-Hot Encoding")
    print("  - Categóricas binarias  -> Ordinal 0/1")
    print("  - Numéricas binarias    -> sin One-Hot")

    with open(output_dir / "feature_groups.json", "w", encoding="utf-8") as f:
        json.dump(asdict(groups), f, indent=2, ensure_ascii=False)

    models = build_models(
        preprocessor=preprocessor,
        y_train_val=y_train_val,
        experiment_set=experiment_set,
    )

    print("\n[INFO] Modelos a evaluar:")
    for name in models:
        print(f"  - {name}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    all_fold_metrics = []
    all_summaries = []

    for model_name, model in models.items():
        try:
            fold_metrics, summary, _, _ = evaluate_cv_model(
                model_name=model_name,
                model=model,
                X=X_train_val,
                y=y_train_val,
                cv=cv,
                output_dir=output_dir,
                threshold_metric="f2",
            )
            all_fold_metrics.append(fold_metrics)
            all_summaries.append(summary)
        except Exception as exc:
            print(f"[ERROR] Falló el modelo {model_name}: {exc}")
            error_path = output_dir / "model_errors.log"
            with open(error_path, "a", encoding="utf-8") as f:
                f.write(f"{model_name}: {repr(exc)}\n")
            continue

    if not all_summaries:
        raise RuntimeError("Todos los modelos fallaron. Revisa dependencias y datos.")

    cv_fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    cv_summary = pd.DataFrame(all_summaries)

    cv_fold_metrics.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    cv_summary = cv_summary.sort_values(["oof_pr_auc", "oof_roc_auc", "oof_f2"], ascending=False)
    cv_summary.to_csv(output_dir / "cv_model_summary.csv", index=False)

    save_comparison_plot(cv_summary, output_dir)

    # -------------------------------------------------------------------------
    # Selección del mejor modelo.
    # `best_row` SIEMPRE debe quedar definido en esta línea antes de usarse más
    # abajo. No renombrar esta variable ni mover la asignación dentro de un
    # try/except o un if: si se hace, cualquier uso posterior (por ejemplo
    # best_row.copy() o best_row["model"]) lanzará NameError porque la variable
    # nunca llegó a crearse. Si necesitas una copia para imprimir o loguear,
    # úsala DESPUÉS de esta línea: best_row_print = best_row.copy().
    # -------------------------------------------------------------------------
    best_row = select_best_model(cv_summary)

    # Validación defensiva: la fila seleccionada debe traer las claves esperadas.
    required_keys = {"model", "best_threshold_oof"}
    missing_keys = required_keys - set(best_row.index)
    if missing_keys:
        raise KeyError(
            f"La fila del mejor modelo no contiene las claves requeridas {sorted(missing_keys)}. "
            f"Claves disponibles: {list(best_row.index)}"
        )

    best_model_name = str(best_row["model"])
    best_threshold = float(best_row["best_threshold_oof"])

    # El nombre seleccionado debe existir en el diccionario de modelos.
    if best_model_name not in models:
        raise KeyError(
            f"El modelo seleccionado '{best_model_name}' no está en el diccionario de modelos "
            f"entrenados. Modelos disponibles: {list(models)}"
        )
    best_model = models[best_model_name]

    print("\n" + "=" * 90)
    print("[OK] Mejor modelo por PR-AUC OOF")
    print("=" * 90)
    # best_row es un pd.Series de tipo MIXTO (contiene el nombre del modelo como
    # texto y el resto como números). best_row.round(4) falla con
    # "TypeError: type str doesn't define __round__" porque intenta redondear el
    # string. Redondeamos solo los valores numéricos y dejamos el texto intacto.
    best_row_print = best_row.copy()
    numeric_mask = best_row_print.map(lambda v: isinstance(v, (int, float))) & best_row_print.notna()
    best_row_print.loc[numeric_mask] = best_row_print.loc[numeric_mask].astype(float).round(4)
    print(best_row_print.to_string())

    final_model, final_metrics = fit_final_and_evaluate(
        best_model_name=best_model_name,
        best_model=best_model,
        best_threshold=best_threshold,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
    )

    joblib.dump(final_model, output_dir / "best_model_pipeline.joblib")

    metadata = {
        "data_path": str(data_path),
        "target": target,
        "id_col": id_col,
        "n_rows": int(len(df)),
        "n_features_input": int(len(selected_features)),
        "features_input": selected_features,
        "drop_cols": drop_cols or [],
        "class_distribution": {str(k): int(v) for k, v in class_counts.to_dict().items()},
        "class_rate": {str(k): float(v) for k, v in class_rate.to_dict().items()},
        "test_size": test_size,
        "n_splits": n_splits,
        "experiment_set": experiment_set,
        "selection_metric": "oof_pr_auc",
        "best_model": best_model_name,
        "best_threshold_oof": best_threshold,
        "feature_groups": asdict(groups),
        "random_state": RANDOM_STATE,
        "outputs": {
            "cv_fold_metrics": "cv_fold_metrics.csv",
            "cv_model_summary": "cv_model_summary.csv",
            "final_test_metrics": "final_test_metrics.csv",
            "best_model_pipeline": "best_model_pipeline.joblib",
            "feature_groups": "feature_groups.json",
        },
    }

    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("[OK] Entrenamiento y selección terminados")
    print(f"[OK] Mejor modelo: {best_model_name}")
    print(f"[OK] Threshold recomendado: {best_threshold:.2f}")
    print(f"[OK] Resumen CV: {output_dir / 'cv_model_summary.csv'}")
    print(f"[OK] Métricas test final: {output_dir / 'final_test_metrics.csv'}")
    print(f"[OK] Pipeline final: {output_dir / 'best_model_pipeline.joblib'}")
    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modelado supervisado unificado con One-Hot, variables binarias, CV e imbalance learning."
    )
    parser.add_argument("--data-path", type=Path, required=True, help="Ruta al dataset final con TARGET sin codificar.")
    parser.add_argument("--target", type=str, default="TARGET", help="Nombre de la columna target.")
    parser.add_argument("--id-col", type=str, default="SK_ID_CURR", help="Columna ID a excluir. Usa '' si no aplica.")
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs/unified_cv_imbalance"), help="Carpeta de salida.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporción para test holdout.")
    parser.add_argument("--n-splits", type=int, default=5, help="Número de folds en validación cruzada.")
    parser.add_argument(
        "--experiment-set",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="quick: modelos principales; full: agrega SMOTE, ADASYN y TomekLinks.",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=None,
        help="Columnas adicionales a excluir del modelado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_col = args.id_col if args.id_col else None

    train_and_select(
        data_path=args.data_path,
        target=args.target,
        id_col=id_col,
        output_dir=args.output_dir,
        drop_cols=args.drop_cols,
        test_size=args.test_size,
        n_splits=args.n_splits,
        experiment_set=args.experiment_set,
    )


if __name__ == "__main__":
    main()
