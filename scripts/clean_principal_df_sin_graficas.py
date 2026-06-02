"""
Limpieza, análisis de importancia y selección final de variables.

Versión ejecutable del notebook `clean_principal_df 1.ipynb`.
Mantiene el mismo flujo y los mismos cálculos del notebook, pero:
- omite las gráficas para que la ejecución no se detenga;
- reemplaza display(...) por impresiones legibles en consola;
- imprime el avance antes y después de los pasos más pesados.

Ejecución recomendada desde la raíz del proyecto:
    python scripts/clean_principal_df_sin_graficas.py

El script espera localizar:
    data/trusted/joined_df.parquet
"""

from datetime import datetime
from pathlib import Path
import gc
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 240)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
RANDOM_STATE = 42

# Umbrales de limpieza y selección. Son los mismos del notebook.
HIGH_NULL_THRESHOLD = 0.90
NEAR_CONSTANT_THRESHOLD = 0.995
HIGH_CARDINALITY_THRESHOLD = 80
OUTLIER_IQR_THRESHOLD = 0.15
CORR_WITH_TARGET_MIN = 0.003
MI_MIN = 0.0001
PERM_IMPORTANCE_MIN = 0.0000
PREDICTOR_CORR_THRESHOLD = 0.90

# Para acelerar el análisis por permutación.
PERMUTATION_VALID_SAMPLE = 20000
PERMUTATION_REPEATS = 5


def log(message: str = "") -> None:
    """Imprime mensajes con hora y libera el buffer para ver avance en tiempo real."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def separator() -> None:
    print("\n" + "=" * 110 + "\n", flush=True)


def print_step(number: int, title: str) -> None:
    separator()
    log(f"PASO {number}: {title}")
    separator()


def show_table(data: pd.DataFrame | pd.Series, title: str | None = None) -> None:
    """Sustituye display(...) del notebook por salida visible en la terminal."""
    if title:
        print(title, flush=True)
    if isinstance(data, pd.Series):
        print(data.to_string(), flush=True)
    elif data.empty:
        print("[Tabla vacía]", flush=True)
    else:
        print(data.to_string(), flush=True)
    print(flush=True)


# ======================================================================================
# PASO 2. Carga del dataset final
# ======================================================================================
def find_joined_dataset(start: Path = Path.cwd()) -> Path:
    """Localiza data/trusted/joined_df.parquet desde el cwd o carpetas padre."""
    checked: list[Path] = []
    for base in [start, *start.parents]:
        candidates = [
            base / "data" / "trusted" / "joined_df.parquet",
            base / "Proyecto_Integrador" / "data" / "trusted" / "joined_df.parquet",
        ]
        for path in candidates:
            checked.append(path)
            if path.exists():
                return path
    checked_txt = "\n  - ".join(str(p) for p in checked)
    raise FileNotFoundError(
        f"No se encontró joined_df.parquet. Rutas revisadas:\n  - {checked_txt}\n"
        "Ejecuta scripts/build_principal_df.py primero."
    )


def load_parquet_safe(path: Path) -> pd.DataFrame:
    """Carga parquet con diagnóstico y menor pico de memoria."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}\n"
            "Verifica que exista joined_df.parquet en la carpeta data/trusted/."
        )

    meta = pq.read_metadata(path)
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"[INFO] Archivo encontrado: {path}", flush=True)
    print(
        f"[INFO] Filas={meta.num_rows:,} | Columnas={meta.num_columns} | "
        f"Tamaño en disco={size_mb:.1f} MB",
        flush=True,
    )

    gc.collect()
    try:
        table = pq.read_table(path, memory_map=True, use_threads=False)
        df_loaded = table.to_pandas()
        del table
    except Exception as exc:
        print(
            f"[WARN] memory_map falló ({type(exc).__name__}). Reintentando lectura estándar...",
            flush=True,
        )
        gc.collect()
        df_loaded = pd.read_parquet(path, engine="pyarrow", use_threads=False)

    for col in df_loaded.select_dtypes(include=["float64"]).columns:
        df_loaded[col] = df_loaded[col].astype("float32")
    for col in df_loaded.select_dtypes(include=["int64"]).columns:
        if col not in {ID_COL, TARGET_COL}:
            df_loaded[col] = pd.to_numeric(df_loaded[col], downcast="integer")

    mem_mb = df_loaded.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[OK] Dataset cargado | RAM estimada ~{mem_mb:.1f} MB", flush=True)
    gc.collect()
    return df_loaded


# ======================================================================================
# PASO 4. Perfil columna por columna
# ======================================================================================
def outlier_iqr_pct(series: pd.Series) -> float:
    """Calcula porcentaje de outliers usando regla IQR para una columna numérica."""
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
    return ((x < lower) | (x > upper)).mean()


def compute_column_profile(data: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    rows = []
    y = data[target_col] if target_col in data.columns else None

    for col in data.columns:
        s = data[col]
        non_null = s.dropna()
        null_count = s.isna().sum()
        null_pct = null_count / len(data)
        nunique = s.nunique(dropna=True)
        nunique_including_null = s.nunique(dropna=False)
        dominant_pct = (
            s.value_counts(dropna=False, normalize=True).iloc[0]
            if len(non_null) > 0
            else 1.0
        )

        is_numeric = pd.api.types.is_numeric_dtype(s)
        outlier_pct = (
            outlier_iqr_pct(s)
            if is_numeric and col not in [target_col, ID_COL]
            else np.nan
        )
        zero_pct = s.eq(0).mean() if is_numeric else np.nan
        skew = (
            pd.to_numeric(s, errors="coerce").skew()
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
                "corr_with_target": corr_target,
                "abs_corr_with_target": abs(corr_target) if pd.notna(corr_target) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["null_pct", "dominant_pct", "abs_corr_with_target"],
        ascending=[False, False, False],
    )


# ======================================================================================
# PASO 7. Indicadores de nulo y tratamiento de outliers
# ======================================================================================
def add_missing_indicators(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    min_missing_pct: float = 0.001,
):
    train_out = train_df.copy()
    valid_out = valid_df.copy()
    missing_cols = train_out.columns[train_out.isna().mean() >= min_missing_pct].tolist()

    for col in missing_cols:
        flag_col = f"{col}_was_missing"
        train_out[flag_col] = train_out[col].isna().astype(int)
        valid_out[flag_col] = valid_out[col].isna().astype(int) if col in valid_out.columns else 0

    return train_out, valid_out, missing_cols


def fit_winsor_limits(train_df: pd.DataFrame, numeric_cols, lower_q=0.01, upper_q=0.99):
    limits = {}
    for col in numeric_cols:
        s = pd.to_numeric(train_df[col], errors="coerce")
        if s.notna().sum() < 10:
            continue
        lower = s.quantile(lower_q)
        upper = s.quantile(upper_q)
        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            limits[col] = (lower, upper)
    return limits


def apply_winsor_limits(data: pd.DataFrame, limits: dict):
    out = data.copy()
    for col, (lower, upper) in limits.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower, upper)
    return out


# ======================================================================================
# PASO 9. Información mutua con TARGET
# ======================================================================================
def prepare_for_mutual_info(X_df: pd.DataFrame):
    X_temp = X_df.copy()
    num_cols = X_temp.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X_temp.columns if c not in num_cols]

    for col in num_cols:
        median = X_temp[col].median()
        X_temp[col] = X_temp[col].fillna(median)

    if cat_cols:
        X_temp[cat_cols] = X_temp[cat_cols].astype("object").fillna("Unknown")
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_temp[cat_cols] = encoder.fit_transform(X_temp[cat_cols])

    discrete_mask = [col in cat_cols or X_temp[col].nunique() <= 20 for col in X_temp.columns]
    return X_temp, discrete_mask


# ======================================================================================
# PASO 11. Redundancia: correlación entre predictores
# ======================================================================================
def choose_feature_to_drop(col_a, col_b, score_map):
    """Entre dos columnas altamente correlacionadas, elimina la de menor score combinado."""
    score_a = score_map.get(col_a, 0)
    score_b = score_map.get(col_b, 0)
    return col_a if score_a < score_b else col_b


def main() -> None:
    total_start = time.perf_counter()

    # ----------------------------------------------------------------------------------
    # 1. Configuración general
    # ----------------------------------------------------------------------------------
    print_step(1, "Configuración general")
    log("Se usarán los mismos umbrales del notebook.")
    print(f"ID_COL={ID_COL} | TARGET_COL={TARGET_COL} | RANDOM_STATE={RANDOM_STATE}", flush=True)
    print(
        f"HIGH_NULL_THRESHOLD={HIGH_NULL_THRESHOLD} | "
        f"NEAR_CONSTANT_THRESHOLD={NEAR_CONSTANT_THRESHOLD} | "
        f"PREDICTOR_CORR_THRESHOLD={PREDICTOR_CORR_THRESHOLD}",
        flush=True,
    )

    # ----------------------------------------------------------------------------------
    # 2. Carga del dataset final
    # ----------------------------------------------------------------------------------
    print_step(2, "Carga del dataset final")
    JOINED_DATASET_PATH = find_joined_dataset()
    DATA_DIR = JOINED_DATASET_PATH.parent.parent  # solo para reportes y export al final
    REPORTS_DIR = DATA_DIR / "feature_selection_reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    size_mb = JOINED_DATASET_PATH.stat().st_size / (1024 ** 2)
    print(f"Dataset a cargar: {JOINED_DATASET_PATH.resolve()} ({size_mb:.1f} MB)", flush=True)

    try:
        df = load_parquet_safe(JOINED_DATASET_PATH)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", flush=True)
        raise
    except Exception as exc:
        if "ArrowMemoryError" in type(exc).__name__ or "MemoryError" in type(exc).__name__:
            print("[ERROR] Memoria insuficiente para cargar el parquet (~500 MB en RAM).", flush=True)
            print("        El archivo SÍ existe; cierra otros notebooks/apps y vuelve a ejecutar.", flush=True)
            print(f"        Detalle: {exc}", flush=True)
        raise

    print(f"Filas: {df.shape[0]:,}", flush=True)
    print(f"Columnas: {df.shape[1]:,}", flush=True)
    show_table(df.head(), "Primeras 5 filas:")

    # ----------------------------------------------------------------------------------
    # 3. Validaciones iniciales
    # ----------------------------------------------------------------------------------
    print_step(3, "Validaciones iniciales")
    required_cols = [ID_COL, TARGET_COL]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Faltan columnas obligatorias: {missing_required}")

    n_duplicated_ids = df[ID_COL].duplicated().sum()
    print(f"Clientes duplicados por {ID_COL}: {n_duplicated_ids:,}", flush=True)

    if n_duplicated_ids > 0:
        print("Se conserva el primer registro por cliente para evitar duplicados en entrenamiento.", flush=True)
        df = df.drop_duplicates(subset=[ID_COL], keep="first")

    target_distribution = df[TARGET_COL].value_counts(dropna=False).to_frame("count")
    target_distribution["pct"] = target_distribution["count"] / len(df)
    show_table(target_distribution, "Distribución de TARGET:")

    if df[TARGET_COL].isna().any():
        raise ValueError("TARGET contiene nulos. Revisa application_train antes de entrenar.")

    # ----------------------------------------------------------------------------------
    # 4. Perfil columna por columna
    # ----------------------------------------------------------------------------------
    print_step(4, "Perfil columna por columna")
    log("Calculando nulos, cardinalidad, valores dominantes, outliers y correlación inicial...")
    profile = compute_column_profile(df)
    profile.to_csv(REPORTS_DIR / "01_column_profile_raw.csv", index=False)

    show_table(profile.head(30), "Primeras 30 filas del perfil de columnas:")
    print("Reporte guardado en:", REPORTS_DIR / "01_column_profile_raw.csv", flush=True)
    print("[OMITIDO] Gráfico: top columnas con mayor porcentaje de nulos.", flush=True)
    print("[OMITIDO] Gráfico: top columnas con más outliers.", flush=True)

    # ----------------------------------------------------------------------------------
    # 5. Reglas básicas de limpieza por columna
    # ----------------------------------------------------------------------------------
    print_step(5, "Reglas básicas de limpieza por columna")
    decisions = []

    def add_decision(column, action, reason):
        decisions.append({"column": column, "action": action, "reason": reason})

    protected_cols = {TARGET_COL}
    initial_drop_cols = {ID_COL}

    for col in initial_drop_cols:
        if col in df.columns:
            add_decision(col, "drop", "Identificador técnico; no debe usarse como predictor.")

    all_null_cols = profile.loc[
        (profile["null_pct"] >= 1.0) & (~profile["column"].isin(protected_cols)),
        "column",
    ].tolist()
    for col in all_null_cols:
        add_decision(col, "drop", "Columna completamente nula.")

    high_null_cols = profile.loc[
        (profile["null_pct"] >= HIGH_NULL_THRESHOLD)
        & (~profile["column"].isin(protected_cols | initial_drop_cols))
        & (~profile["column"].isin(all_null_cols)),
        "column",
    ].tolist()
    for col in high_null_cols:
        add_decision(
            col,
            "drop",
            f"Tiene >= {HIGH_NULL_THRESHOLD:.0%} de nulos; baja estabilidad para entrenamiento.",
        )

    constant_cols = profile.loc[
        (profile["nunique_including_null"] <= 1)
        & (~profile["column"].isin(protected_cols | initial_drop_cols)),
        "column",
    ].tolist()
    for col in constant_cols:
        add_decision(col, "drop", "Columna constante; no aporta separación entre clientes.")

    near_constant_cols = profile.loc[
        (profile["dominant_pct"] >= NEAR_CONSTANT_THRESHOLD)
        & (~profile["column"].isin(protected_cols | initial_drop_cols))
        & (~profile["column"].isin(constant_cols))
        & (~profile["column"].isin(all_null_cols))
        & (~profile["column"].isin(high_null_cols)),
        "column",
    ].tolist()
    for col in near_constant_cols:
        add_decision(col, "drop", f"Casi constante: un valor domina >= {NEAR_CONSTANT_THRESHOLD:.1%}.")

    quality_drop_cols = sorted(
        set(initial_drop_cols)
        | set(all_null_cols)
        | set(high_null_cols)
        | set(constant_cols)
        | set(near_constant_cols)
    )

    gc.collect()
    drop_cols = [c for c in quality_drop_cols if c in df.columns]
    clean_base = df.drop(columns=drop_cols, errors="ignore")
    clean_base.replace([np.inf, -np.inf], np.nan, inplace=True)

    # df solo conserva columnas que pueden llegar al export final.
    keep_cols = [ID_COL] + [c for c in clean_base.columns if c != ID_COL]
    df = df[keep_cols]
    gc.collect()

    print(f"Columnas iniciales: {len(drop_cols) + clean_base.shape[1]:,}", flush=True)
    print(f"Columnas eliminadas por reglas básicas: {len(quality_drop_cols):,}", flush=True)
    print(f"Columnas después de limpieza básica: {clean_base.shape[1]:,}", flush=True)

    quality_decisions = pd.DataFrame(decisions)
    quality_decisions.to_csv(REPORTS_DIR / "02_basic_cleaning_decisions.csv", index=False)
    show_table(quality_decisions.head(100), "Primeras decisiones de limpieza:")
    print("Reporte guardado en:", REPORTS_DIR / "02_basic_cleaning_decisions.csv", flush=True)

    # ----------------------------------------------------------------------------------
    # 6. Separación train/valid para evitar leakage
    # ----------------------------------------------------------------------------------
    print_step(6, "Separación train/valid para evitar leakage")
    X = clean_base.drop(columns=[TARGET_COL])
    y = clean_base[TARGET_COL].astype(int)

    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"X_train_raw: {X_train_raw.shape}", flush=True)
    print(f"X_valid_raw: {X_valid_raw.shape}", flush=True)
    print(f"Target rate train: {y_train.mean():.4f}", flush=True)
    print(f"Target rate valid: {y_valid.mean():.4f}", flush=True)

    del clean_base, X, y
    gc.collect()

    # ----------------------------------------------------------------------------------
    # 7. Indicadores de nulo y tratamiento de outliers
    # ----------------------------------------------------------------------------------
    print_step(7, "Indicadores de nulo y tratamiento de outliers")
    log("Creando indicadores de nulo y calculando límites winsorizados P1-P99 solo con train...")
    X_train_miss, X_valid_miss, missing_indicator_source_cols = add_missing_indicators(
        X_train_raw, X_valid_raw
    )

    numeric_cols_before = X_train_miss.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    winsor_limits = fit_winsor_limits(X_train_miss, numeric_cols_before)

    X_train_clean = apply_winsor_limits(X_train_miss, winsor_limits)
    X_valid_clean = apply_winsor_limits(X_valid_miss, winsor_limits)

    print(f"Columnas con indicador de nulo creado: {len(missing_indicator_source_cols):,}", flush=True)
    print(f"Columnas numéricas con límites de outliers P1-P99: {len(winsor_limits):,}", flush=True)

    # ----------------------------------------------------------------------------------
    # 8. Correlación de variables numéricas con TARGET
    # ----------------------------------------------------------------------------------
    print_step(8, "Correlación de variables numéricas con TARGET")
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

    corr_target = pd.DataFrame(corr_rows).sort_values("abs_pearson_corr_target", ascending=False)
    corr_target.to_csv(REPORTS_DIR / "03_numeric_correlation_with_target.csv", index=False)

    show_table(corr_target.head(30), "Top 30 variables numéricas por correlación con TARGET:")
    print("Reporte guardado en:", REPORTS_DIR / "03_numeric_correlation_with_target.csv", flush=True)
    print("[OMITIDO] Gráfico: top variables numéricas por correlación absoluta con TARGET.", flush=True)

    # ----------------------------------------------------------------------------------
    # 9. Información mutua con TARGET
    # ----------------------------------------------------------------------------------
    print_step(9, "Información mutua con TARGET")
    log("Preparando variables para información mutua...")
    X_mi, discrete_mask = prepare_for_mutual_info(X_train_clean)

    log("Calculando mutual_info_classif. Este paso puede tardar...")
    mi_values = mutual_info_classif(
        X_mi,
        y_train,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )
    log("Cálculo de información mutua finalizado.")

    mi_report = pd.DataFrame(
        {"column": X_mi.columns, "mutual_info_target": mi_values}
    ).sort_values("mutual_info_target", ascending=False)
    mi_report.to_csv(REPORTS_DIR / "04_mutual_information_with_target.csv", index=False)

    show_table(mi_report.head(40), "Top 40 variables por información mutua:")
    print("Reporte guardado en:", REPORTS_DIR / "04_mutual_information_with_target.csv", flush=True)
    print("[OMITIDO] Gráfico: top variables por información mutua.", flush=True)

    # ----------------------------------------------------------------------------------
    # 10. Modelo base e importancia por permutación
    # ----------------------------------------------------------------------------------
    print_step(10, "Modelo base e importancia por permutación")
    numeric_features = X_train_clean.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X_train_clean.columns if c not in numeric_features]

    print(f"Features numéricas: {len(numeric_features):,}", flush=True)
    print(f"Features categóricas: {len(categorical_features):,}", flush=True)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", ohe),
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

    model_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    log("Entrenando RandomForestClassifier base. Este paso puede tardar...")
    model_pipeline.fit(X_train_clean, y_train)
    log("Modelo base entrenado.")
    valid_pred = model_pipeline.predict_proba(X_valid_clean)[:, 1]
    valid_auc = roc_auc_score(y_valid, valid_pred)
    print(f"AUC validación modelo base: {valid_auc:.4f}", flush=True)

    if len(X_valid_clean) > PERMUTATION_VALID_SAMPLE:
        valid_sample_idx = X_valid_clean.sample(
            PERMUTATION_VALID_SAMPLE, random_state=RANDOM_STATE
        ).index
        X_perm = X_valid_clean.loc[valid_sample_idx]
        y_perm = y_valid.loc[valid_sample_idx]
    else:
        X_perm = X_valid_clean
        y_perm = y_valid

    log(
        f"Calculando importancia por permutación con {len(X_perm):,} filas "
        f"y {PERMUTATION_REPEATS} repeticiones. Este paso puede tardar..."
    )
    perm_result = permutation_importance(
        model_pipeline,
        X_perm,
        y_perm,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    log("Importancia por permutación calculada.")

    perm_report = pd.DataFrame(
        {
            "column": X_perm.columns,
            "perm_importance_mean": perm_result.importances_mean,
            "perm_importance_std": perm_result.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)

    perm_report.to_csv(REPORTS_DIR / "05_permutation_importance.csv", index=False)
    show_table(perm_report.head(40), "Top 40 variables por importancia por permutación:")
    print("Reporte guardado en:", REPORTS_DIR / "05_permutation_importance.csv", flush=True)
    print("[OMITIDO] Gráfico: top variables por importancia en validación.", flush=True)

    del model_pipeline, X_perm, y_perm, X_valid_clean, X_train_miss, X_valid_miss
    del X_train_raw, X_valid_raw
    gc.collect()

    # ----------------------------------------------------------------------------------
    # 11. Redundancia: correlación entre predictores
    # ----------------------------------------------------------------------------------
    print_step(11, "Redundancia: correlación entre predictores")
    importance_report = profile[
        ["column", "null_pct", "dominant_pct", "outlier_iqr_pct", "nunique"]
    ].copy()
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

    for col in [
        "abs_pearson_corr_target",
        "abs_spearman_corr_target",
        "mutual_info_target",
        "perm_importance_mean",
    ]:
        importance_report[col] = importance_report[col].fillna(0)

    importance_report["combined_score"] = (
        importance_report["abs_pearson_corr_target"].rank(pct=True)
        + importance_report["abs_spearman_corr_target"].rank(pct=True)
        + importance_report["mutual_info_target"].rank(pct=True)
        + importance_report["perm_importance_mean"].rank(pct=True)
    )

    importance_report = importance_report.sort_values("combined_score", ascending=False)
    importance_report.to_csv(REPORTS_DIR / "06_combined_feature_importance.csv", index=False)
    show_table(importance_report.head(50), "Top 50 variables por score combinado:")

    candidate_cols = [c for c in X_train_clean.columns if c in importance_report["column"].values]
    score_map = dict(zip(importance_report["column"], importance_report["combined_score"]))

    numeric_candidate_cols = [c for c in candidate_cols if c in numeric_features]
    base_keep_signal = importance_report.set_index("column").loc[numeric_candidate_cols]
    numeric_for_corr = base_keep_signal[
        (base_keep_signal["abs_pearson_corr_target"] >= CORR_WITH_TARGET_MIN)
        | (base_keep_signal["mutual_info_target"] >= MI_MIN)
        | (base_keep_signal["perm_importance_mean"] > PERM_IMPORTANCE_MIN)
    ].index.tolist()

    corr_drop_cols = set()
    high_corr_pairs = []

    if len(numeric_for_corr) >= 2:
        corr_matrix = X_train_clean[numeric_for_corr].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        for col_b in upper.columns:
            correlated_a = upper.index[upper[col_b] >= PREDICTOR_CORR_THRESHOLD].tolist()
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
        else pd.DataFrame()
    )
    corr_pairs_report.to_csv(REPORTS_DIR / "07_highly_correlated_pairs.csv", index=False)

    print(f"Pares con correlación >= {PREDICTOR_CORR_THRESHOLD}: {len(high_corr_pairs):,}", flush=True)
    print(f"Columnas a eliminar por redundancia: {len(corr_drop_cols):,}", flush=True)
    show_table(corr_pairs_report.head(50), "Primeros pares de variables altamente correlacionadas:")

    # ----------------------------------------------------------------------------------
    # 12. Selección final de columnas importantes
    # ----------------------------------------------------------------------------------
    print_step(12, "Selección final de columnas importantes")
    remaining_cols_after_quality = [c for c in X_train_clean.columns if c not in quality_drop_cols]
    importance_lookup = importance_report.set_index("column")

    selected_features = []
    selection_rows = []

    for col in remaining_cols_after_quality:
        if col not in importance_lookup.index:
            continue

        row = importance_lookup.loc[col]
        is_categorical = col in categorical_features
        has_target_signal = (
            row["abs_pearson_corr_target"] >= CORR_WITH_TARGET_MIN
            or row["abs_spearman_corr_target"] >= CORR_WITH_TARGET_MIN
            or row["mutual_info_target"] >= MI_MIN
            or row["perm_importance_mean"] > PERM_IMPORTANCE_MIN
        )

        keep = bool(has_target_signal or is_categorical)
        reason = []
        if row["abs_pearson_corr_target"] >= CORR_WITH_TARGET_MIN:
            reason.append("correlación Pearson con TARGET")
        if row["abs_spearman_corr_target"] >= CORR_WITH_TARGET_MIN:
            reason.append("correlación Spearman con TARGET")
        if row["mutual_info_target"] >= MI_MIN:
            reason.append("información mutua")
        if row["perm_importance_mean"] > PERM_IMPORTANCE_MIN:
            reason.append("importancia por permutación")
        if is_categorical and not reason:
            reason.append("categórica no descartada por calidad")

        if col in corr_drop_cols:
            keep = False
            reason.append("eliminada por alta correlación con otra variable más fuerte")

        if keep:
            selected_features.append(col)

        selection_rows.append(
            {
                "column": col,
                "selected": keep,
                "reason": "; ".join(reason) if reason else "sin señal suficiente frente al target",
                "abs_pearson_corr_target": row["abs_pearson_corr_target"],
                "abs_spearman_corr_target": row["abs_spearman_corr_target"],
                "mutual_info_target": row["mutual_info_target"],
                "perm_importance_mean": row["perm_importance_mean"],
                "combined_score": row["combined_score"],
            }
        )

    selection_report = pd.DataFrame(selection_rows).sort_values(
        ["selected", "combined_score"], ascending=[False, False]
    )
    selection_report.to_csv(REPORTS_DIR / "08_final_feature_selection_report.csv", index=False)

    print(f"Features candidatas después de limpieza básica: {len(remaining_cols_after_quality):,}", flush=True)
    print(f"Features seleccionadas finales: {len(selected_features):,}", flush=True)
    print(
        f"Features descartadas finales: {len(remaining_cols_after_quality) - len(selected_features):,}",
        flush=True,
    )

    show_table(selection_report.head(80), "Primeras 80 decisiones de selección final:")
    print("Reporte guardado en:", REPORTS_DIR / "08_final_feature_selection_report.csv", flush=True)

    # ----------------------------------------------------------------------------------
    # Contexto del negocio y reducción de variables: eliminación manual del notebook
    # ----------------------------------------------------------------------------------
    print_step(13, "Reducción manual de variables según criterio de negocio")
    COLUMNAS_ELIMINADAS = [
        "bureau_min_days_credit",
        "bureau_avg_days_update",
        "bureau_n_card",
        "bureau_mean_days_overrun",
        "prev_application_mean",
        "prev_annuity_approved_mean",
        "prev_cnt_payment_max",
        "prev_insured_on_approval_count",
        "prev_insured_on_approval_rate",
        "WEEKDAY_APPR_PROCESS_START",
        "FONDKAPREMONT_MODE",
        "NAME_TYPE_SUITE",
        "FLAG_OWN_REALTY",
        "EMERGENCYSTATE_MODE",
        "WALLSMATERIAL_MODE",
        "HOUSETYPE_MODE",
        "LANDAREA_AVG",
        "NONLIVINGAREA_AVG",
        "BASEMENTAREA_AVG",
        "CNT_FAM_MEMBERS",
        "INCOME_PER_PERSON",
        "OBS_60_CNT_SOCIAL_CIRCLE",
    ]

    cols_antes = len(selected_features)
    presentes = [c for c in COLUMNAS_ELIMINADAS if c in selected_features]
    ausentes = [c for c in COLUMNAS_ELIMINADAS if c not in selected_features]

    selected_features = [c for c in selected_features if c not in COLUMNAS_ELIMINADAS]
    cols_despues = len(selected_features)

    mask = selection_report["column"].isin(presentes)
    selection_report.loc[mask, "selected"] = False
    selection_report.loc[mask, "reason"] = "eliminada manualmente por redundancia o bajo aporte de negocio"

    print(f"Features seleccionadas antes : {cols_antes:,}", flush=True)
    print(f"Eliminadas                   : {len(presentes):,}", flush=True)
    print(f"Features seleccionadas después: {cols_despues:,}", flush=True)

    if presentes:
        print("\nColumnas removidas del set final:", flush=True)
        for col in presentes:
            print(f"  - {col}", flush=True)

    if ausentes:
        print(f"\n[AVISO] {len(ausentes)} columnas no estaban en selected_features:", flush=True)
        for col in ausentes:
            print(f"  - {col}", flush=True)

    # ----------------------------------------------------------------------------------
    # Agrupación de categorías poco frecuentes de ORGANIZATION_TYPE
    # ----------------------------------------------------------------------------------
    print_step(14, "Agrupación de categorías poco frecuentes de ORGANIZATION_TYPE")
    ORG_COL = "ORGANIZATION_TYPE"
    MIN_COUNT = 1974

    if ORG_COL not in df.columns or ORG_COL not in selected_features:
        print(f"[AVISO] {ORG_COL} no está en el dataset final.", flush=True)
    else:
        org_clean = (
            df[ORG_COL]
            .astype("object")
            .replace("XNA", "unknown")
            .fillna("Missing")
        )

        counts = org_clean.value_counts()
        low_cats = set(counts[counts < MIN_COUNT].index)
        org_grouped = org_clean.where(~org_clean.isin(low_cats), "others")

        org_dist = org_grouped.value_counts(dropna=False).rename("cantidad").to_frame()
        org_dist["porcentaje"] = (org_dist["cantidad"] / len(df) * 100).round(2)
        org_dist = org_dist.sort_values("cantidad", ascending=False)

        df[ORG_COL] = org_grouped.values

        print(f"Distribución de {ORG_COL} en el dataset final", flush=True)
        print(f"Filas: {len(df):,} | Categorías: {len(org_dist):,}", flush=True)
        print(f"XNA -> unknown | categorías con < {MIN_COUNT:,} registros -> others\n", flush=True)
        show_table(org_dist)

    # ----------------------------------------------------------------------------------
    # 13. Dataset final para entrenamiento
    # ----------------------------------------------------------------------------------
    print_step(15, "Dataset final para entrenamiento")
    TRUSTED_DIR = JOINED_DATASET_PATH.parent
    SELECTED_DATASET_PATH = TRUSTED_DIR / "train_selected_features.parquet"

    del X_train_clean, profile, importance_report, corr_target, mi_report, perm_report
    gc.collect()

    final_cols = [ID_COL, TARGET_COL] + selected_features
    final_train_selected = df[final_cols]
    final_train_selected = final_train_selected.replace([np.inf, -np.inf], np.nan)

    assert final_train_selected[ID_COL].is_unique, "SK_ID_CURR no es único en el dataset final."
    assert final_train_selected[TARGET_COL].notna().all(), "TARGET contiene nulos."
    assert final_train_selected.shape[0] == df.shape[0], "Se perdieron filas durante la selección."

    TRUSTED_DIR.mkdir(parents=True, exist_ok=True)
    final_train_selected.to_parquet(SELECTED_DATASET_PATH, index=False)

    selected_features_path = REPORTS_DIR / "selected_features.txt"
    selected_features_path.write_text("\n".join(selected_features), encoding="utf-8")

    selected_categorical_features = [c for c in selected_features if c in categorical_features]

    print("Dataset final seleccionado", flush=True)
    print(f"Filas: {final_train_selected.shape[0]:,}", flush=True)
    print(
        f"Columnas totales incluyendo ID y TARGET: {final_train_selected.shape[1]:,}",
        flush=True,
    )
    print(f"Features para entrenamiento: {len(selected_features):,}", flush=True)
    print(f"Categóricas que quedaron: {len(selected_categorical_features):,}", flush=True)
    for col in selected_categorical_features:
        print(f"  - {col}", flush=True)
    print(f"Guardado en: {SELECTED_DATASET_PATH.resolve()}", flush=True)
    print(f"Lista de features guardada en: {selected_features_path.resolve()}", flush=True)

    show_table(final_train_selected.head(), "Primeras 5 filas del dataset final:")

    # ----------------------------------------------------------------------------------
    # 14. Pipeline recomendado para entrenar con las columnas seleccionadas
    # ----------------------------------------------------------------------------------
    print_step(16, "Pipeline recomendado para entrenar con las columnas seleccionadas")
    X_selected = final_train_selected[selected_features]
    y_selected = final_train_selected[TARGET_COL].astype(int)

    selected_numeric_features = X_selected.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    selected_categorical_features = [
        c for c in X_selected.columns if c not in selected_numeric_features
    ]

    try:
        selected_ohe = OneHotEncoder(
            handle_unknown="ignore", min_frequency=0.01, sparse_output=False
        )
    except TypeError:
        selected_ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse=False)

    selected_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), selected_numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", selected_ohe),
                    ]
                ),
                selected_categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    print(f"Numéricas seleccionadas: {len(selected_numeric_features):,}", flush=True)
    print(f"Categóricas seleccionadas: {len(selected_categorical_features):,}", flush=True)
    print("Preprocesador listo para conectarse con el modelo final.", flush=True)

    # ----------------------------------------------------------------------------------
    # 15. Resumen ejecutivo de decisiones
    # ----------------------------------------------------------------------------------
    print_step(17, "Resumen ejecutivo de decisiones")
    print("Reportes generados en:", REPORTS_DIR.resolve(), flush=True)
    print("  - 01_column_profile_raw.csv", flush=True)
    print("  - 02_basic_cleaning_decisions.csv", flush=True)
    print("  - 03_numeric_correlation_with_target.csv", flush=True)
    print("  - 04_mutual_information_with_target.csv", flush=True)
    print("  - 05_permutation_importance.csv", flush=True)
    print("  - 06_combined_feature_importance.csv", flush=True)
    print("  - 07_highly_correlated_pairs.csv", flush=True)
    print("  - 08_final_feature_selection_report.csv", flush=True)
    print("  - selected_features.txt", flush=True)
    print("\nSalida principal:", SELECTED_DATASET_PATH.resolve(), flush=True)

    elapsed = time.perf_counter() - total_start
    separator()
    log(f"PROCESO FINALIZADO CORRECTAMENTE en {elapsed / 60:.2f} minutos.")
    separator()

    # Se mantienen estas referencias para uso interactivo si se importa el script.
    _ = y_selected, selected_preprocessor


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        separator()
        log(f"[ERROR] La ejecución se detuvo: {type(exc).__name__}: {exc}")
        separator()
        raise
