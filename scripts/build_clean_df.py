"""
Construye clean_df a partir de joined_df aplicando las reglas del notebook
clean_principal_df.ipynb.

Entrada por defecto:
    data/trusted/joined_df.parquet

Salidas por defecto:
    data/trusted/clean_df.parquet
    data/trusted/reports/feature_selection_report.csv

El script conserva una fila por SK_ID_CURR, mantiene SK_ID_CURR y TARGET para
trazabilidad/entrenamiento, y deja solo las variables seleccionadas para modelado.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
RANDOM_STATE = 42

# Umbrales definidos en el notebook.
HIGH_NULL_THRESHOLD = 0.80
NEAR_CONSTANT_THRESHOLD = 0.90
HIGH_CARDINALITY_THRESHOLD = 80  # Se conserva como referencia; el notebook no lo usa para descartar.
OUTLIER_IQR_THRESHOLD = 0.15     # Se conserva como referencia; el notebook perfila outliers y winsoriza P1-P99.
CORR_WITH_TARGET_MIN = 0.003
MI_MIN = 0.0001
PERM_IMPORTANCE_MIN = 0.0000
PREDICTOR_CORR_THRESHOLD = 0.90

PERMUTATION_VALID_SAMPLE = 20_000
PERMUTATION_REPEATS = 5


def find_data_dir(start: Optional[Path] = None) -> Path:
    """Busca una carpeta data/ desde el directorio actual o sus padres."""
    start = Path.cwd() if start is None else Path(start)
    for base in [start, *start.parents]:
        candidates = [base / "data", base / "Proyecto_Integrador" / "data"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "No se encontró carpeta data/. Ejecuta el script desde el proyecto "
        "o pasa rutas explícitas con --input/--output/--report."
    )


def outlier_iqr_pct(series: pd.Series) -> float:
    """Calcula porcentaje de outliers con la regla IQR para una columna numérica."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 10:
        return np.nan
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((x < lower) | (x > upper)).mean())


def compute_column_profile(data: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Perfila cada columna para justificar reglas de limpieza y selección."""
    rows = []
    y = data[target_col] if target_col in data.columns else None

    for col in data.columns:
        s = data[col]
        non_null = s.dropna()
        null_count = int(s.isna().sum())
        null_pct = float(null_count / len(data)) if len(data) else 0.0
        nunique = int(s.nunique(dropna=True))
        nunique_including_null = int(s.nunique(dropna=False))
        dominant_pct = (
            float(s.value_counts(dropna=False, normalize=True).iloc[0])
            if len(non_null) > 0
            else 1.0
        )

        is_numeric = bool(pd.api.types.is_numeric_dtype(s))
        outlier_pct = outlier_iqr_pct(s) if is_numeric and col not in [target_col, ID_COL] else np.nan
        zero_pct = float(s.eq(0).mean()) if is_numeric else np.nan
        skew = (
            float(pd.to_numeric(s, errors="coerce").skew())
            if is_numeric and col not in [target_col, ID_COL]
            else np.nan
        )

        corr_target = np.nan
        if y is not None and is_numeric and col not in [target_col, ID_COL]:
            valid = s.notna() & y.notna()
            if valid.sum() > 10 and s[valid].nunique() > 1:
                corr_target = s[valid].corr(y[valid])

        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "is_numeric": is_numeric,
                "null_count": null_count,
                "null_pct": null_pct,
                "nunique": nunique,
                "nunique_including_null": nunique_including_null,
                "dominant_pct": dominant_pct,
                "zero_pct": zero_pct,
                "outlier_iqr_pct": outlier_pct,
                "skew": skew,
                "corr_with_target_raw": corr_target,
                "abs_corr_with_target_raw": abs(corr_target) if pd.notna(corr_target) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["null_pct", "dominant_pct", "abs_corr_with_target_raw"],
        ascending=[False, False, False],
    )


def build_quality_decisions(
    profile: pd.DataFrame,
    high_null_threshold: float = HIGH_NULL_THRESHOLD,
    near_constant_threshold: float = NEAR_CONSTANT_THRESHOLD,
) -> Tuple[List[str], pd.DataFrame]:
    """Aplica las reglas básicas de descarte por calidad definidas en el notebook."""
    decisions = []

    def add_decision(column: str, action: str, reason: str) -> None:
        decisions.append({"column": column, "action": action, "reason": reason})

    protected_cols = {TARGET_COL}
    initial_drop_cols = {ID_COL}

    for col in initial_drop_cols:
        if col in profile["column"].values:
            add_decision(col, "drop", "Identificador técnico; no debe usarse como predictor.")

    all_null_cols = profile.loc[
        (profile["null_pct"] >= 1.0) & (~profile["column"].isin(protected_cols)),
        "column",
    ].tolist()
    for col in all_null_cols:
        add_decision(col, "drop", "Columna completamente nula.")

    high_null_cols = profile.loc[
        (profile["null_pct"] >= high_null_threshold)
        & (~profile["column"].isin(protected_cols | initial_drop_cols))
        & (~profile["column"].isin(all_null_cols)),
        "column",
    ].tolist()
    for col in high_null_cols:
        add_decision(
            col,
            "drop",
            f"Tiene >= {high_null_threshold:.0%} de nulos; baja estabilidad para entrenamiento.",
        )

    constant_cols = profile.loc[
        (profile["nunique_including_null"] <= 1)
        & (~profile["column"].isin(protected_cols | initial_drop_cols)),
        "column",
    ].tolist()
    for col in constant_cols:
        add_decision(col, "drop", "Columna constante; no aporta separación entre clientes.")

    near_constant_cols = profile.loc[
        (profile["dominant_pct"] >= near_constant_threshold)
        & (~profile["column"].isin(protected_cols | initial_drop_cols))
        & (~profile["column"].isin(constant_cols))
        & (~profile["column"].isin(all_null_cols))
        & (~profile["column"].isin(high_null_cols)),
        "column",
    ].tolist()
    for col in near_constant_cols:
        add_decision(col, "drop", f"Casi constante: un valor domina >= {near_constant_threshold:.1%}.")

    quality_drop_cols = sorted(
        set(initial_drop_cols)
        | set(all_null_cols)
        | set(high_null_cols)
        | set(constant_cols)
        | set(near_constant_cols)
    )

    return quality_drop_cols, pd.DataFrame(decisions)


def add_missing_indicators(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    min_missing_pct: float = 0.001,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Crea indicadores de nulo ajustados con train y aplicados a valid."""
    train_out = train_df.copy()
    valid_out = valid_df.copy()
    missing_cols = train_out.columns[train_out.isna().mean() >= min_missing_pct].tolist()

    for col in missing_cols:
        flag_col = f"{col}_was_missing"
        train_out[flag_col] = train_out[col].isna().astype(int)
        valid_out[flag_col] = valid_out[col].isna().astype(int) if col in valid_out.columns else 0

    return train_out, valid_out, missing_cols


def add_missing_indicators_to_full(data: pd.DataFrame, missing_cols: Sequence[str]) -> pd.DataFrame:
    """Aplica al dataset completo los indicadores definidos usando train."""
    out = data.copy()
    for col in missing_cols:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def fit_winsor_limits(
    train_df: pd.DataFrame,
    numeric_cols: Iterable[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Dict[str, Tuple[float, float]]:
    """Ajusta límites P1-P99 con train para evitar leakage."""
    limits = {}
    for col in numeric_cols:
        s = pd.to_numeric(train_df[col], errors="coerce")
        if s.notna().sum() < 10:
            continue
        lower = s.quantile(lower_q)
        upper = s.quantile(upper_q)
        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            limits[col] = (float(lower), float(upper))
    return limits


def apply_winsor_limits(data: pd.DataFrame, limits: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Recorta variables numéricas con límites preajustados."""
    out = data.copy()
    for col, (lower, upper) in limits.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower, upper)
    return out


def fit_imputation_values(train_df: pd.DataFrame) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Ajusta imputadores simples con train: mediana para numéricas, Unknown para categóricas."""
    numeric_cols = train_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in train_df.columns if c not in numeric_cols]

    medians = {}
    for col in numeric_cols:
        median = pd.to_numeric(train_df[col], errors="coerce").median()
        medians[col] = float(median) if pd.notna(median) else 0.0

    return medians, numeric_cols, categorical_cols


def apply_imputation(
    data: pd.DataFrame,
    medians: Dict[str, float],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    """Aplica imputación al dataset completo usando valores ajustados con train."""
    out = data.copy()
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(medians.get(col, 0.0))
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].astype("object").fillna("Unknown")
    return out


def compute_numeric_target_correlations(X_train_clean: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Calcula correlación Pearson y Spearman de numéricas contra TARGET."""
    numeric_cols = X_train_clean.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    corr_rows = []

    for col in numeric_cols:
        s = pd.to_numeric(X_train_clean[col], errors="coerce")
        valid = s.notna() & y_train.notna()
        if valid.sum() > 20 and s[valid].nunique() > 1:
            pearson_corr = s[valid].corr(y_train[valid], method="pearson")
            spearman_corr = s[valid].corr(y_train[valid], method="spearman")
        else:
            pearson_corr = np.nan
            spearman_corr = np.nan

        corr_rows.append(
            {
                "column": col,
                "pearson_corr_target": pearson_corr,
                "spearman_corr_target": spearman_corr,
                "abs_pearson_corr_target": abs(pearson_corr) if pd.notna(pearson_corr) else np.nan,
                "abs_spearman_corr_target": abs(spearman_corr) if pd.notna(spearman_corr) else np.nan,
            }
        )

    return pd.DataFrame(corr_rows).sort_values("abs_pearson_corr_target", ascending=False)


def prepare_for_mutual_info(X_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[bool]]:
    """Prepara numéricas y categóricas para mutual_info_classif."""
    X_temp = X_df.copy()
    num_cols = X_temp.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X_temp.columns if c not in num_cols]

    for col in num_cols:
        median = X_temp[col].median()
        X_temp[col] = X_temp[col].fillna(median if pd.notna(median) else 0)

    if cat_cols:
        X_temp[cat_cols] = X_temp[cat_cols].astype("object").fillna("Unknown")
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_temp[cat_cols] = encoder.fit_transform(X_temp[cat_cols])

    discrete_mask = [col in cat_cols or X_temp[col].nunique() <= 20 for col in X_temp.columns]
    return X_temp, discrete_mask


def compute_mutual_information(X_train_clean: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Calcula información mutua contra TARGET."""
    if X_train_clean.empty:
        return pd.DataFrame(columns=["column", "mutual_info_target"])

    X_mi, discrete_mask = prepare_for_mutual_info(X_train_clean)
    mi_values = mutual_info_classif(
        X_mi,
        y_train,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )
    return pd.DataFrame({"column": X_mi.columns, "mutual_info_target": mi_values}).sort_values(
        "mutual_info_target", ascending=False
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """Compatibilidad entre versiones de scikit-learn para sparse/sparse_output."""
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse=False)


def compute_permutation_importance_report(
    X_train_clean: pd.DataFrame,
    X_valid_clean: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    permutation_valid_sample: int = PERMUTATION_VALID_SAMPLE,
    permutation_repeats: int = PERMUTATION_REPEATS,
) -> Tuple[pd.DataFrame, float]:
    """Entrena un RandomForest base y calcula importancia por permutación."""
    numeric_features = X_train_clean.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_clean.columns if c not in numeric_features]

    if not numeric_features and not categorical_features:
        empty = pd.DataFrame(columns=["column", "perm_importance_mean", "perm_importance_std"])
        return empty, np.nan

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    base_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", base_model)])
    model_pipeline.fit(X_train_clean, y_train)
    valid_pred = model_pipeline.predict_proba(X_valid_clean)[:, 1]
    valid_auc = float(roc_auc_score(y_valid, valid_pred))

    if len(X_valid_clean) > permutation_valid_sample:
        valid_sample_idx = X_valid_clean.sample(permutation_valid_sample, random_state=RANDOM_STATE).index
        X_perm = X_valid_clean.loc[valid_sample_idx]
        y_perm = y_valid.loc[valid_sample_idx]
    else:
        X_perm = X_valid_clean
        y_perm = y_valid

    perm_result = permutation_importance(
        model_pipeline,
        X_perm,
        y_perm,
        scoring="roc_auc",
        n_repeats=permutation_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    perm_report = pd.DataFrame(
        {
            "column": X_perm.columns,
            "perm_importance_mean": perm_result.importances_mean,
            "perm_importance_std": perm_result.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)

    return perm_report, valid_auc


def build_importance_report(
    profile: pd.DataFrame,
    corr_target: pd.DataFrame,
    mi_report: pd.DataFrame,
    perm_report: pd.DataFrame,
) -> pd.DataFrame:
    """Combina métricas de calidad e importancia en un score único."""
    base_cols = ["column", "dtype", "is_numeric", "null_pct", "dominant_pct", "outlier_iqr_pct", "nunique"]

    # El notebook calcula importancia sobre X_train_clean, que puede contener
    # columnas creadas como *_was_missing. Por eso el reporte parte de la unión
    # entre columnas originales y columnas evaluadas por correlación/MI/permutación.
    all_cols = list(
        dict.fromkeys(
            profile["column"].tolist()
            + corr_target.get("column", pd.Series(dtype=str)).tolist()
            + mi_report.get("column", pd.Series(dtype=str)).tolist()
            + perm_report.get("column", pd.Series(dtype=str)).tolist()
        )
    )
    importance_report = pd.DataFrame({"column": all_cols})
    importance_report = importance_report.merge(
        profile[[c for c in base_cols if c in profile.columns]],
        on="column",
        how="left",
    )

    importance_report = importance_report.merge(
        corr_target[
            [
                "column",
                "pearson_corr_target",
                "abs_pearson_corr_target",
                "spearman_corr_target",
                "abs_spearman_corr_target",
            ]
        ],
        on="column",
        how="left",
    )
    importance_report = importance_report.merge(mi_report, on="column", how="left")
    importance_report = importance_report.merge(perm_report, on="column", how="left")

    metric_cols = [
        "abs_pearson_corr_target",
        "abs_spearman_corr_target",
        "mutual_info_target",
        "perm_importance_mean",
    ]
    for col in metric_cols:
        importance_report[col] = importance_report[col].fillna(0)

    importance_report["combined_score"] = (
        importance_report["abs_pearson_corr_target"].rank(pct=True)
        + importance_report["abs_spearman_corr_target"].rank(pct=True)
        + importance_report["mutual_info_target"].rank(pct=True)
        + importance_report["perm_importance_mean"].rank(pct=True)
    )

    return importance_report.sort_values("combined_score", ascending=False)


def choose_feature_to_drop(col_a: str, col_b: str, score_map: Dict[str, float]) -> str:
    """Entre dos columnas altamente correlacionadas, elimina la de menor score combinado."""
    score_a = score_map.get(col_a, 0)
    score_b = score_map.get(col_b, 0)
    return col_a if score_a < score_b else col_b


def find_correlated_drop_columns(
    X_train_clean: pd.DataFrame,
    importance_report: pd.DataFrame,
    numeric_features: Sequence[str],
    corr_with_target_min: float = CORR_WITH_TARGET_MIN,
    mi_min: float = MI_MIN,
    perm_importance_min: float = PERM_IMPORTANCE_MIN,
    predictor_corr_threshold: float = PREDICTOR_CORR_THRESHOLD,
) -> Tuple[set, pd.DataFrame]:
    """Identifica variables numéricas redundantes por alta correlación entre predictores."""
    candidate_cols = [c for c in X_train_clean.columns if c in importance_report["column"].values]
    score_map = dict(zip(importance_report["column"], importance_report["combined_score"]))

    numeric_candidate_cols = [c for c in candidate_cols if c in numeric_features]
    if not numeric_candidate_cols:
        return set(), pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr", "dropped", "kept"])

    base_keep_signal = importance_report.set_index("column").reindex(numeric_candidate_cols)
    numeric_for_corr = base_keep_signal[
        (base_keep_signal["abs_pearson_corr_target"] >= corr_with_target_min)
        | (base_keep_signal["mutual_info_target"] >= mi_min)
        | (base_keep_signal["perm_importance_mean"] > perm_importance_min)
    ].index.tolist()

    corr_drop_cols = set()
    high_corr_pairs = []

    if len(numeric_for_corr) >= 2:
        corr_matrix = X_train_clean[numeric_for_corr].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        for col_b in upper.columns:
            correlated_a = upper.index[upper[col_b] >= predictor_corr_threshold].tolist()
            for col_a in correlated_a:
                drop_col = choose_feature_to_drop(col_a, col_b, score_map)
                keep_col = col_b if drop_col == col_a else col_a
                corr_drop_cols.add(drop_col)
                high_corr_pairs.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "abs_corr": upper.loc[col_a, col_b],
                        "dropped": drop_col,
                        "kept": keep_col,
                    }
                )

    corr_pairs_report = (
        pd.DataFrame(high_corr_pairs).sort_values("abs_corr", ascending=False)
        if high_corr_pairs
        else pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr", "dropped", "kept"])
    )
    return corr_drop_cols, corr_pairs_report


def build_selection_report(
    original_profile: pd.DataFrame,
    importance_report: pd.DataFrame,
    quality_decisions: pd.DataFrame,
    quality_drop_cols: Sequence[str],
    X_train_clean: pd.DataFrame,
    categorical_features: Sequence[str],
    corr_drop_cols: set,
    missing_indicator_source_cols: Sequence[str],
    winsor_limits: Dict[str, Tuple[float, float]],
    valid_auc: float,
) -> Tuple[List[str], pd.DataFrame]:
    """Selecciona features finales y genera un reporte de justificación columna a columna."""
    quality_reason_map = (
        quality_decisions.drop_duplicates("column", keep="first").set_index("column")["reason"].to_dict()
        if not quality_decisions.empty
        else {}
    )
    quality_drop_set = set(quality_drop_cols)
    importance_lookup = importance_report.set_index("column")

    selected_features = []
    rows = []

    # Incluye columnas originales y columnas creadas por indicadores de nulo.
    all_report_cols = list(dict.fromkeys(list(original_profile["column"]) + list(X_train_clean.columns)))

    original_profile_map = original_profile.set_index("column").to_dict(orient="index")

    for col in all_report_cols:
        is_id = col == ID_COL
        is_target = col == TARGET_COL
        is_created_missing_indicator = col.endswith("_was_missing") and col in X_train_clean.columns
        source_missing_col = col[: -len("_was_missing")] if is_created_missing_indicator else ""
        was_winsorized = col in winsor_limits

        if is_id:
            rows.append(
                {
                    "column": col,
                    "role": "id",
                    "selected": True,
                    "action": "keep_as_id",
                    "reason": "Se conserva para trazabilidad, pero no se usa como predictor.",
                    "created_from": "",
                    "was_winsorized_p1_p99": False,
                    "valid_auc_model_base": valid_auc,
                }
            )
            continue

        if is_target:
            rows.append(
                {
                    "column": col,
                    "role": "target",
                    "selected": True,
                    "action": "keep_as_target",
                    "reason": "Variable objetivo supervisada; se conserva para entrenamiento.",
                    "created_from": "",
                    "was_winsorized_p1_p99": False,
                    "valid_auc_model_base": valid_auc,
                }
            )
            continue

        if col in quality_drop_set:
            rows.append(
                {
                    "column": col,
                    "role": "predictor",
                    "selected": False,
                    "action": "drop_quality",
                    "reason": quality_reason_map.get(col, "Descartada por reglas básicas de calidad."),
                    "created_from": source_missing_col,
                    "was_winsorized_p1_p99": was_winsorized,
                    "valid_auc_model_base": valid_auc,
                }
            )
            continue

        if col not in importance_lookup.index:
            # Esto puede ocurrir con columnas presentes solo en reportes/raw o descartadas antes de importancia.
            rows.append(
                {
                    "column": col,
                    "role": "predictor",
                    "selected": False,
                    "action": "drop_not_evaluated",
                    "reason": "No quedó disponible como predictor evaluable después de la limpieza básica.",
                    "created_from": source_missing_col,
                    "was_winsorized_p1_p99": was_winsorized,
                    "valid_auc_model_base": valid_auc,
                }
            )
            continue

        row = importance_lookup.loc[col]
        is_categorical = col in categorical_features
        has_target_signal = bool(
            row["abs_pearson_corr_target"] >= CORR_WITH_TARGET_MIN
            or row["abs_spearman_corr_target"] >= CORR_WITH_TARGET_MIN
            or row["mutual_info_target"] >= MI_MIN
            or row["perm_importance_mean"] > PERM_IMPORTANCE_MIN
        )

        keep = bool(has_target_signal or is_categorical)
        reason_parts = []
        if row["abs_pearson_corr_target"] >= CORR_WITH_TARGET_MIN:
            reason_parts.append("correlación Pearson con TARGET")
        if row["abs_spearman_corr_target"] >= CORR_WITH_TARGET_MIN:
            reason_parts.append("correlación Spearman con TARGET")
        if row["mutual_info_target"] >= MI_MIN:
            reason_parts.append("información mutua")
        if row["perm_importance_mean"] > PERM_IMPORTANCE_MIN:
            reason_parts.append("importancia por permutación")
        if is_categorical and not reason_parts:
            reason_parts.append("categórica no descartada por calidad")
        if is_created_missing_indicator:
            reason_parts.append(f"indicador de nulo creado desde {source_missing_col}")

        if col in corr_drop_cols:
            keep = False
            reason_parts.append("eliminada por alta correlación con otra variable más fuerte")

        if keep:
            selected_features.append(col)
            action = "keep_feature"
        elif col in corr_drop_cols:
            action = "drop_redundant_correlation"
        else:
            action = "drop_low_signal"

        rows.append(
            {
                "column": col,
                "role": "predictor",
                "selected": keep,
                "action": action,
                "reason": "; ".join(reason_parts) if reason_parts else "sin señal suficiente frente al target",
                "created_from": source_missing_col,
                "was_winsorized_p1_p99": was_winsorized,
                "valid_auc_model_base": valid_auc,
            }
        )

    selection_report = pd.DataFrame(rows)

    # Agrega métricas de perfil raw e importancia.
    profile_metrics = original_profile.copy()
    selection_report = selection_report.merge(profile_metrics, on="column", how="left")

    importance_cols = [
        "column",
        "pearson_corr_target",
        "abs_pearson_corr_target",
        "spearman_corr_target",
        "abs_spearman_corr_target",
        "mutual_info_target",
        "perm_importance_mean",
        "perm_importance_std",
        "combined_score",
    ]
    selection_report = selection_report.merge(
        importance_report[[c for c in importance_cols if c in importance_report.columns]],
        on="column",
        how="left",
    )

    selection_report = selection_report.sort_values(
        ["selected", "role", "combined_score"],
        ascending=[False, True, False],
        na_position="last",
    )

    return selected_features, selection_report


def clean_and_select_features(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    permutation_valid_sample: int = PERMUTATION_VALID_SAMPLE,
    permutation_repeats: int = PERMUTATION_REPEATS,
) -> pd.DataFrame:
    """Ejecuta el flujo completo de limpieza y selección de variables."""
    print(f"[INFO] Leyendo dataset unido: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el dataset de entrada: {input_path}")

    df = pd.read_parquet(input_path)
    print(f"[OK] Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")

    required_cols = [ID_COL, TARGET_COL]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Faltan columnas obligatorias: {missing_required}")

    n_duplicated_ids = int(df[ID_COL].duplicated().sum())
    if n_duplicated_ids > 0:
        print(f"[AVISO] {n_duplicated_ids:,} clientes duplicados por {ID_COL}. Se conserva el primer registro.")
        df = df.drop_duplicates(subset=[ID_COL], keep="first").copy()

    if df[TARGET_COL].isna().any():
        raise ValueError("TARGET contiene nulos. Revisa el dataset unido antes de limpiar.")

    # Perfil y reglas básicas.
    profile = compute_column_profile(df)
    quality_drop_cols, quality_decisions = build_quality_decisions(profile)

    clean_base = df.drop(columns=[c for c in quality_drop_cols if c in df.columns], errors="ignore").copy()
    clean_base = clean_base.replace([np.inf, -np.inf], np.nan)

    print(f"[INFO] Columnas eliminadas por calidad: {len(quality_drop_cols):,}")
    print(f"[INFO] Columnas después de limpieza básica: {clean_base.shape[1]:,}")

    X = clean_base.drop(columns=[TARGET_COL])
    y = clean_base[TARGET_COL].astype(int)

    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Indicadores de nulos, winsorización y métricas de selección.
    X_train_miss, X_valid_miss, missing_indicator_source_cols = add_missing_indicators(X_train_raw, X_valid_raw)

    numeric_cols_before = X_train_miss.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    winsor_limits = fit_winsor_limits(X_train_miss, numeric_cols_before)

    X_train_clean = apply_winsor_limits(X_train_miss, winsor_limits)
    X_valid_clean = apply_winsor_limits(X_valid_miss, winsor_limits)

    corr_target = compute_numeric_target_correlations(X_train_clean, y_train)
    mi_report = compute_mutual_information(X_train_clean, y_train)
    perm_report, valid_auc = compute_permutation_importance_report(
        X_train_clean,
        X_valid_clean,
        y_train,
        y_valid,
        permutation_valid_sample=permutation_valid_sample,
        permutation_repeats=permutation_repeats,
    )

    importance_report = build_importance_report(profile, corr_target, mi_report, perm_report)

    numeric_features = X_train_clean.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_clean.columns if c not in numeric_features]
    corr_drop_cols, _corr_pairs_report = find_correlated_drop_columns(
        X_train_clean,
        importance_report,
        numeric_features,
    )

    selected_features, selection_report = build_selection_report(
        original_profile=profile,
        importance_report=importance_report,
        quality_decisions=quality_decisions,
        quality_drop_cols=quality_drop_cols,
        X_train_clean=X_train_clean,
        categorical_features=categorical_features,
        corr_drop_cols=corr_drop_cols,
        missing_indicator_source_cols=missing_indicator_source_cols,
        winsor_limits=winsor_limits,
        valid_auc=valid_auc,
    )

    # Aplica al dataset completo las transformaciones ajustadas sobre train.
    X_full = clean_base.drop(columns=[TARGET_COL]).copy()
    X_full = add_missing_indicators_to_full(X_full, missing_indicator_source_cols)
    X_full = apply_winsor_limits(X_full, winsor_limits)

    medians, full_numeric_cols, full_categorical_cols = fit_imputation_values(X_train_clean)
    X_full = apply_imputation(X_full, medians, full_numeric_cols, full_categorical_cols)

    missing_selected = [c for c in selected_features if c not in X_full.columns]
    if missing_selected:
        raise RuntimeError(
            "Algunas variables seleccionadas no existen en el dataset limpio completo: "
            f"{missing_selected[:20]}"
        )

    clean_df = pd.concat(
        [df[[ID_COL, TARGET_COL]].reset_index(drop=True), X_full[selected_features].reset_index(drop=True)],
        axis=1,
    )

    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)

    assert clean_df[ID_COL].is_unique, f"{ID_COL} no es único en el dataset final."
    assert clean_df[TARGET_COL].notna().all(), "TARGET contiene nulos."
    assert clean_df.shape[0] == df.shape[0], "Se perdieron filas durante la selección."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_parquet(output_path, index=False)
    selection_report.to_csv(report_path, index=False)

    print("=" * 80)
    print("[OK] Dataset limpio construido")
    print(f"Filas finales                : {clean_df.shape[0]:,}")
    print(f"Columnas finales             : {clean_df.shape[1]:,}")
    print(f"Features seleccionadas        : {len(selected_features):,}")
    print(f"Indicadores de nulo creados   : {len(missing_indicator_source_cols):,}")
    print(f"Variables winsorizadas P1-P99 : {len(winsor_limits):,}")
    print(f"AUC validación modelo base    : {valid_auc:.4f}" if pd.notna(valid_auc) else "AUC validación modelo base    : N/A")
    print(f"Dataset guardado en           : {output_path}")
    print(f"Reporte guardado en           : {report_path}")
    print("=" * 80)

    return clean_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpia joined_df.parquet, selecciona variables relevantes y guarda clean_df + reporte."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Ruta del dataset unido de entrada. Default: data/trusted/joined_df.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del dataset limpio de salida. Default: data/trusted/clean_df.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Ruta del reporte de justificación. Default: data/trusted/reports/feature_selection_report.csv",
    )
    parser.add_argument(
        "--permutation-valid-sample",
        type=int,
        default=PERMUTATION_VALID_SAMPLE,
        help="Máximo de filas de validación para permutation importance. Default: 20000.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=PERMUTATION_REPEATS,
        help="Repeticiones para permutation importance. Default: 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is None or args.output is None or args.report is None:
        data_dir = find_data_dir()
        trusted_dir = data_dir / "trusted"
        if args.input is None:
            args.input = trusted_dir / "joined_df.parquet"
        if args.output is None:
            args.output = trusted_dir / "clean_df.parquet"
        if args.report is None:
            args.report = trusted_dir / "reports" / "feature_selection_report.csv"

    clean_and_select_features(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        permutation_valid_sample=args.permutation_valid_sample,
        permutation_repeats=args.permutation_repeats,
    )


if __name__ == "__main__":
    main()