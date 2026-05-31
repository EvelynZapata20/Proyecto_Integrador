"""
Entrenamiento básico para clasificación binaria con desbalanceo 90/10.

Uso típico:
    python 09_basic_imbalanced_training.py \
        --data-path data/train_selected_features.parquet.parquet \
        --target TARGET \
        --id-col SK_ID_CURR \
        --output-dir model_outputs

Qué hace:
    1. Lee el dataset final ya unido.
    2. Hace split estratificado train / validation / test.
    3. Entrena modelos básicos con class_weight para manejar desbalanceo.
    4. Elige umbral en validation optimizando F2-score.
    5. Evalúa en test con ROC-AUC, PR-AUC, precision, recall, F1, F2, balanced accuracy y matriz de confusión.
    6. Guarda métricas, curvas y el mejor pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


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


def load_selected_features(
    df_columns: List[str],
    target: str,
    id_col: str | None,
) -> List[str]:
    """
    Usa todas las columnas disponibles del dataset como variables predictoras,
    excluyendo target e id.

    Este dataset ya debe venir filtrado con las variables seleccionadas.
    """
    excluded = {target}

    if id_col:
        excluded.add(id_col)

    features = [c for c in df_columns if c not in excluded]

    print(f"[INFO] Variables disponibles para entrenar: {len(features):,}")

    if not features:
        raise ValueError("No quedó ninguna variable disponible para entrenar.")

    return features


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatibilidad entre versiones de scikit-learn: sparse_output vs sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """Modelos básicos con manejo de desbalanceo por class_weight."""
    models = {
        "dummy_prior": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", DummyClassifier(strategy="prior", random_state=RANDOM_STATE)),
            ]
        ),
        "logistic_balanced": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="saga",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=50,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    return models


def get_positive_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Devuelve probabilidad de clase positiva TARGET=1."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Conversión simple a escala 0-1 para métricas de ranking.
        return 1 / (1 + np.exp(-scores))
    raise ValueError("El modelo no tiene predict_proba ni decision_function.")


def metrics_at_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
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
    y_true: pd.Series,
    y_prob: np.ndarray,
    metric: str = "f2",
) -> Tuple[float, pd.DataFrame]:
    """
    Busca umbral en validation.
    F2 prioriza recall de TARGET=1, útil cuando la clase minoritaria es riesgo/default.
    """
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    rows = []
    for thr in thresholds:
        rows.append(metrics_at_threshold(y_true, y_prob, thr))

    table = pd.DataFrame(rows)
    best_idx = table[metric].idxmax()
    return float(table.loc[best_idx, "threshold"]), table


def save_curves(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    """Guarda curva ROC y Precision-Recall."""
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
    plt.savefig(output_dir / f"roc_curve_{model_name}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"pr_curve_{model_name}.png", dpi=150)
    plt.close()


def train_and_evaluate(
    data_path: Path,
    target: str,
    id_col: str | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset(data_path)
    print(f"[INFO] Dataset leído: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")

    if target not in df.columns:
        raise ValueError(f"No existe la columna target: {target}")

    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)

    class_counts = df[target].value_counts().sort_index()
    class_rate = df[target].value_counts(normalize=True).sort_index()
    print("\n[INFO] Distribución del target:")
    for cls in class_counts.index:
        print(f"  Clase {cls}: {class_counts.loc[cls]:,} ({class_rate.loc[cls]:.2%})")

    DROP_BUSINESS_COLS = [
        "OWN_CAR_AGE",
        "FLAG_OWN_CAR"
    ]

    selected_features = load_selected_features(
        df_columns=df.columns.tolist(),
        target=target,
        id_col=id_col,
    )

    dropped_existing = [col for col in DROP_BUSINESS_COLS if col in selected_features]
    dropped_missing = [col for col in DROP_BUSINESS_COLS if col not in selected_features]

    selected_features = [
        col for col in selected_features
        if col not in DROP_BUSINESS_COLS
    ]

    print(f"[INFO] Variables eliminadas por negocio: {len(dropped_existing)}")
    print(f"       {dropped_existing}")

    if dropped_missing:
        print(f"[AVISO] Variables a eliminar que no estaban en selected_features: {dropped_missing}")

    print(f"[INFO] Variables finales para entrenar: {len(selected_features):,}")

    X = df[selected_features].copy()
    y = df[target].copy()

    # Split honesto: train / validation / test estratificado.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,  # 0.25 de 80% = 20% total
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print("\n[INFO] Split:")
    print(f"  Train     : {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test      : {len(X_test):,}")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    print("\n[INFO] Tipos de variables:")
    print(f"  Numéricas   : {len(numeric_cols):,}")
    print(f"  Categóricas : {len(categorical_cols):,}")

    models = build_models(preprocessor)

    all_rows = []
    threshold_tables = {}
    fitted_models = {}

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"[INFO] Entrenando: {model_name}")
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        val_prob = get_positive_proba(model, X_val)
        test_prob = get_positive_proba(model, X_test)

        # Umbral default 0.50 y umbral ajustado en validation.
        best_threshold, threshold_table = find_best_threshold(y_val, val_prob, metric="f2")
        threshold_tables[model_name] = threshold_table

        val_default = metrics_at_threshold(y_val, val_prob, 0.50)
        val_best = metrics_at_threshold(y_val, val_prob, best_threshold)
        test_default = metrics_at_threshold(y_test, test_prob, 0.50)
        test_best = metrics_at_threshold(y_test, test_prob, best_threshold)

        for split_name, threshold_name, row in [
            ("validation", "default_0_50", val_default),
            ("validation", "best_f2_threshold", val_best),
            ("test", "default_0_50", test_default),
            ("test", "best_f2_threshold", test_best),
        ]:
            row = row.copy()
            row["model"] = model_name
            row["split"] = split_name
            row["threshold_type"] = threshold_name
            all_rows.append(row)

        print(f"[INFO] Mejor threshold por F2 en validation: {best_threshold:.2f}")
        print("\n[TEST] Métricas con threshold ajustado:")
        print(pd.Series(test_best).round(4).to_string())

        y_test_pred = (test_prob >= best_threshold).astype(int)
        report_txt = classification_report(y_test, y_test_pred, digits=4, zero_division=0)
        with open(output_dir / f"classification_report_{model_name}.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)

        save_curves(y_test, test_prob, model_name, output_dir)

    metrics_df = pd.DataFrame(all_rows)
    order_cols = ["model", "split", "threshold_type"] + [c for c in metrics_df.columns if c not in {"model", "split", "threshold_type"}]
    metrics_df = metrics_df[order_cols]
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    for model_name, table in threshold_tables.items():
        table.to_csv(output_dir / f"threshold_search_{model_name}.csv", index=False)

    # Elegimos mejor modelo por PR-AUC en validation; PR-AUC es más informativo que accuracy con 90/10.
    val_rank = metrics_df[
        (metrics_df["split"].eq("validation"))
        & (metrics_df["threshold_type"].eq("best_f2_threshold"))
        & (~metrics_df["model"].eq("dummy_prior"))
    ].sort_values(["pr_auc", "roc_auc", "f2"], ascending=False)

    best_model_name = val_rank.iloc[0]["model"]
    best_threshold = float(val_rank.iloc[0]["threshold"])
    best_model = fitted_models[best_model_name]

    joblib.dump(best_model, output_dir / "best_model_pipeline.joblib")

    metadata = {
        "best_model": best_model_name,
        "best_threshold": best_threshold,
        "target": target,
        "id_col": id_col,
        "n_features": len(selected_features),
        "features": selected_features,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "random_state": RANDOM_STATE,
        "selection_rule": "todas las columnas menos target/id",
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("[OK] Entrenamiento terminado")
    print(f"[OK] Mejor modelo según validation PR-AUC: {best_model_name}")
    print(f"[OK] Threshold recomendado: {best_threshold:.2f}")
    print(f"[OK] Métricas guardadas en: {output_dir / 'metrics_summary.csv'}")
    print(f"[OK] Pipeline guardado en: {output_dir / 'best_model_pipeline.joblib'}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento básico con desbalanceo 90/10.")
    parser.add_argument("--data-path", type=Path, required=True, help="Ruta al dataset final con TARGET.")
    parser.add_argument("--target", type=str, default="TARGET", help="Nombre de la columna target.")
    parser.add_argument("--id-col", type=str, default="SK_ID_CURR", help="Columna ID a excluir.")
    parser.add_argument("--output-dir", type=Path, default=Path("model_outputs"), help="Carpeta de salida.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        data_path=args.data_path,
        target=args.target,
        id_col=args.id_col,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
