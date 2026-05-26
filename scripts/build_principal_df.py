"""
Este script toma application_train como tabla principal y le une features
agregadas de las tablas históricas, dejando una sola fila por SK_ID_CURR.

Notas:
    - Todas las tablas históricas se agregan antes del join.
    - El join contra application_train siempre es LEFT JOIN para no perder clientes.
    - Los clientes sin historial en alguna tabla quedan marcados con flags `no_*_history`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Utilidades generales
# =============================================================================

def find_data_dir(start: Optional[Path] = None) -> Path:
    """
    Busca una carpeta data/ desde el directorio actual o sus padres.
    También contempla Proyecto_Integrador/data/.
    """
    start = Path.cwd() if start is None else Path(start)
    for base in [start, *start.parents]:
        candidates = [
            base / "data",
            base / "Proyecto_Integrador" / "data",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "No se encontró carpeta data/. Ejecuta el script desde el proyecto "
        "o pasa --data-dir con la ruta correcta."
    )


def read_table(data_dir: Path, stem: str, required: bool = True) -> Optional[pd.DataFrame]:
    """
    Lee una tabla usando un nombre base. Intenta parquet y csv.
    También soporta CSVs descomprimidos como carpeta:
        data/credit_card_balance.csv/credit_card_balance.csv
    """
    candidates = [
        data_dir / f"{stem}.parquet",
        data_dir / f"{stem}.csv",
        data_dir / f"{stem}.csv" / f"{stem}.csv",
        data_dir / stem / f"{stem}.csv",
        data_dir / stem / f"{stem}.parquet",
    ]

    # Algunos archivos del proyecto pueden venir con nombres alternativos.
    aliases = {
        "pos_cash_balance": [
            data_dir / "POS_CASH_balance.parquet",
            data_dir / "POS_CASH_balance.csv",
            data_dir / "POS_CASH_balance.csv" / "POS_CASH_balance.csv",
        ],
        "POS_CASH_balance": [
            data_dir / "pos_cash_balance.parquet",
            data_dir / "pos_cash_balance.csv",
            data_dir / "pos_cash_balance.csv" / "pos_cash_balance.csv",
        ],
    }
    candidates.extend(aliases.get(stem, []))

    for path in candidates:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                print(f"[OK] Leyendo {path}")
                return pd.read_parquet(path)
            if path.suffix.lower() == ".csv":
                print(f"[OK] Leyendo {path}")
                return pd.read_csv(path)

    if required:
        checked = "\n  - ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No se encontró la tabla {stem}. Rutas revisadas:\n  - {checked}")

    print(f"[AVISO] No se encontró {stem}; se omitirá.")
    return None


def safe_div(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Divide evitando infinitos y denominadores en cero."""
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def ensure_unique_by_id(df: pd.DataFrame, id_col: str, name: str) -> pd.DataFrame:
    """
    Garantiza una fila por id. Si hay duplicados, conserva el primero.
    Esto no debería pasar en application_train, pero protege el flujo final.
    """
    duplicated = df[id_col].duplicated().sum()
    if duplicated:
        print(f"[AVISO] {name} tenía {duplicated:,} ids duplicados. Se conserva el primer registro.")
        df = df.drop_duplicates(subset=[id_col], keep="first").copy()
    return df


def left_join_features(
    base: pd.DataFrame,
    features: Optional[pd.DataFrame],
    id_col: str,
    table_name: str,
    history_flag_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Une features agregadas a la tabla principal.
    Luego marca ausencia de historial y rellena NaN de esas features con 0.
    """
    if features is None or features.empty:
        print(f"[AVISO] Features de {table_name} vacías; se omite join.")
        return base

    features = ensure_unique_by_id(features, id_col, table_name)
    feature_cols = [c for c in features.columns if c != id_col]

    before_rows = len(base)
    base = base.merge(features, on=id_col, how="left")

    if len(base) != before_rows:
        raise RuntimeError(
            f"El join de {table_name} cambió el número de filas "
            f"({before_rows:,} -> {len(base):,}). Revisa duplicados."
        )

    if history_flag_col:
        base[history_flag_col] = base[feature_cols[0]].isna().astype(int)

    base[feature_cols] = base[feature_cols].fillna(0)
    print(f"[OK] Join {table_name}: +{len(feature_cols)} features")
    return base


# =============================================================================
# application_train: limpieza y features base
# =============================================================================

def clean_application_train(app: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce las reglas principales del notebook de limpieza application_train:

    - Elimina housing MODE/MEDI por alta correlación con AVG.
    - Elimina FLAG_DOCUMENT con tasa menor a 0.5%.
    - Elimina variables casi constantes.
    - Elimina variables redundantes por correlación alta.
    - Trata el valor anómalo DAYS_EMPLOYED = 365243.
    - Imputa EXT_SOURCE, housing, consultas bureau, social circle y montos.
    - Crea ratios financieros y variables derivadas.
    """
    app = app.copy()

    housing_base = [
        "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD",
        "COMMONAREA", "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN",
        "LANDAREA", "LIVINGAPARTMENTS", "LIVINGAREA",
        "NONLIVINGAPARTMENTS", "NONLIVINGAREA",
    ]

    # 1) Housing: conservar AVG y eliminar MODE/MEDI.
    cols_housing_drop = []
    for base in housing_base:
        for suffix in ["_MODE", "_MEDI"]:
            col = f"{base}{suffix}"
            if col in app.columns:
                cols_housing_drop.append(col)
    app = app.drop(columns=cols_housing_drop, errors="ignore")

    # 2) Documentos de muy baja frecuencia.
    doc_cols = [c for c in app.columns if c.startswith("FLAG_DOCUMENT")]
    low_doc = [c for c in doc_cols if app[c].mean() < 0.005]
    app = app.drop(columns=low_doc, errors="ignore")

    # 3) Variables casi constantes.
    app = app.drop(columns=["FLAG_MOBIL", "FLAG_CONT_MOBILE"], errors="ignore")

    # 4) Variables redundantes por correlación.
    app = app.drop(columns=["REGION_RATING_CLIENT"], errors="ignore")
    app = app.drop(columns=["OBS_30_CNT_SOCIAL_CIRCLE"], errors="ignore")

    # 5) DAYS_EMPLOYED: 365243 marca no empleo / pensionado en Home Credit.
    if "DAYS_EMPLOYED" in app.columns:
        app["DAYS_EMPLOYED_FLAG"] = (app["DAYS_EMPLOYED"] == 365243).astype(int)
        app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)
        app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].fillna(app["DAYS_EMPLOYED"].median())

    # 6) EXT_SOURCE: imputar con mediana.
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if col in app.columns:
            app[col] = app[col].fillna(app[col].median())

    # 7) Housing AVG: ausencia informativa.
    housing_avg_cols = [
        c for c in app.columns
        if any(c.startswith(base) for base in housing_base) and c.endswith("_AVG")
    ]
    for col in housing_avg_cols:
        app[col] = app[col].fillna(-1)

    for col in ["FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]:
        if col in app.columns:
            app[col] = app[col].fillna("Unknown")

    if "TOTALAREA_MODE" in app.columns:
        app["TOTALAREA_MODE"] = app["TOTALAREA_MODE"].fillna(-1)

    # 8) Consultas al bureau: nulo como ausencia de consulta registrada.
    bureau_req_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    for col in bureau_req_cols:
        if col in app.columns:
            app[col] = app[col].fillna(0)

    # 9) Imputaciones específicas.
    if "OCCUPATION_TYPE" in app.columns:
        app["OCCUPATION_TYPE"] = app["OCCUPATION_TYPE"].fillna("Unknown")

    if "NAME_TYPE_SUITE" in app.columns and app["NAME_TYPE_SUITE"].isna().any():
        mode = app["NAME_TYPE_SUITE"].mode(dropna=True)
        app["NAME_TYPE_SUITE"] = app["NAME_TYPE_SUITE"].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    for col in ["OBS_60_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"]:
        if col in app.columns:
            app[col] = app[col].fillna(app[col].median())

    for col in ["AMT_GOODS_PRICE", "AMT_ANNUITY", "CNT_FAM_MEMBERS", "DAYS_LAST_PHONE_CHANGE"]:
        if col in app.columns:
            app[col] = app[col].fillna(app[col].median())

    if "OWN_CAR_AGE" in app.columns:
        app["OWN_CAR_AGE"] = app["OWN_CAR_AGE"].fillna(0)

    # 10) Feature engineering application.
    if "DAYS_BIRTH" in app.columns:
        app["AGE_YEARS"] = app["DAYS_BIRTH"] / -365.25
        app = app.drop(columns=["DAYS_BIRTH"], errors="ignore")

    if "DAYS_EMPLOYED" in app.columns:
        app["YEARS_EMPLOYED"] = app["DAYS_EMPLOYED"] / -365.25

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(app.columns):
        app["CREDIT_INCOME_RATIO"] = safe_div(app["AMT_CREDIT"], app["AMT_INCOME_TOTAL"])

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(app.columns):
        app["ANNUITY_INCOME_RATIO"] = safe_div(app["AMT_ANNUITY"], app["AMT_INCOME_TOTAL"])

    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(app.columns):
        app["CREDIT_GOODS_RATIO"] = safe_div(app["AMT_CREDIT"], app["AMT_GOODS_PRICE"])
        app["CREDIT_GOODS_RATIO"] = app["CREDIT_GOODS_RATIO"].fillna(app["CREDIT_GOODS_RATIO"].median())

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(app.columns):
        app["INCOME_PER_PERSON"] = safe_div(app["AMT_INCOME_TOTAL"], app["CNT_FAM_MEMBERS"])
        app["INCOME_PER_PERSON"] = app["INCOME_PER_PERSON"].fillna(app["INCOME_PER_PERSON"].median())

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in app.columns]
    if ext_cols:
        app["EXT_SOURCE_MEAN"] = app[ext_cols].mean(axis=1)
        app["EXT_SOURCE_MIN"] = app[ext_cols].min(axis=1)

    existing_bureau_req_cols = [c for c in bureau_req_cols if c in app.columns]
    if existing_bureau_req_cols:
        app["BUREAU_QUERIES_TOTAL"] = app[existing_bureau_req_cols].sum(axis=1)

    return ensure_unique_by_id(app, "SK_ID_CURR", "application_train")


# =============================================================================
# bureau: agregación por SK_ID_CURR
# =============================================================================

def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Features del historial crediticio externo.

    Reglas tomadas de los notebooks:
    - AMT_ANNUITY y AMT_CREDIT_MAX_OVERDUE se eliminan como variables raw por alto nulo.
    - AMT_CREDIT_SUM nulo se elimina.
    - AMT_CREDIT_SUM_LIMIT y AMT_CREDIT_SUM_DEBT se imputan con 0.
    - DAYS_CREDIT_ENDDATE se imputa con mediana.
    - Se agrega a nivel SK_ID_CURR para hacer join con application_train.
    """
    bur = bureau.copy()

    bur = bur.drop(columns=["AMT_ANNUITY", "AMT_CREDIT_MAX_OVERDUE", "CREDIT_CURRENCY"], errors="ignore")

    if "AMT_CREDIT_SUM" in bur.columns:
        bur = bur.dropna(subset=["AMT_CREDIT_SUM"])

    for col in ["AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE"]:
        if col in bur.columns:
            bur[col] = bur[col].fillna(0)

    if "DAYS_CREDIT_ENDDATE" in bur.columns:
        bur["DAYS_CREDIT_ENDDATE"] = bur["DAYS_CREDIT_ENDDATE"].fillna(bur["DAYS_CREDIT_ENDDATE"].median())

    group = bur.groupby("SK_ID_CURR")

    features = pd.DataFrame(index=group.size().index)
    features["bureau_n_credits"] = group["SK_ID_BUREAU"].count()
    features["bureau_n_active"] = group["CREDIT_ACTIVE"].apply(lambda x: x.eq("Active").sum())
    features["bureau_n_closed"] = group["CREDIT_ACTIVE"].apply(lambda x: x.eq("Closed").sum())
    features["bureau_n_bad"] = group["CREDIT_ACTIVE"].apply(lambda x: x.isin(["Bad debt", "Sold"]).sum())
    features["bureau_avg_days_credit"] = group["DAYS_CREDIT"].mean()
    features["bureau_min_days_credit"] = group["DAYS_CREDIT"].min()
    features["bureau_avg_days_update"] = group["DAYS_CREDIT_UPDATE"].mean()

    features["bureau_max_overdue_days"] = group["CREDIT_DAY_OVERDUE"].max()
    features["bureau_sum_overdue_days"] = group["CREDIT_DAY_OVERDUE"].sum()

    features["bureau_total_credit"] = group["AMT_CREDIT_SUM"].sum()
    features["bureau_total_debt"] = group["AMT_CREDIT_SUM_DEBT"].sum()
    features["bureau_total_overdue_amt"] = group["AMT_CREDIT_SUM_OVERDUE"].sum()
    features["bureau_n_prolonged"] = group["CNT_CREDIT_PROLONG"].sum()

    features["bureau_n_consumer"] = group["CREDIT_TYPE"].apply(lambda x: x.eq("Consumer credit").sum())
    features["bureau_n_card"] = group["CREDIT_TYPE"].apply(lambda x: x.eq("Credit card").sum())
    features["bureau_n_mortgage"] = group["CREDIT_TYPE"].apply(lambda x: x.eq("Mortgage").sum())
    features["bureau_n_car_loan"] = group["CREDIT_TYPE"].apply(lambda x: x.eq("Car loan").sum())

    features["bureau_active_pct"] = safe_div(features["bureau_n_active"], features["bureau_n_credits"])
    features["bureau_closed_pct"] = safe_div(features["bureau_n_closed"], features["bureau_n_credits"])
    features["bureau_has_bad_debt"] = (features["bureau_n_bad"] > 0).astype(int)
    features["bureau_has_overdue"] = (features["bureau_max_overdue_days"] > 0).astype(int)
    features["bureau_debt_ratio"] = safe_div(features["bureau_total_debt"], features["bureau_total_credit"].abs() + 1)

    # Features de créditos cerrados: duración y desfase vs fecha esperada.
    required_closed_cols = {"CREDIT_ACTIVE", "DAYS_ENDDATE_FACT", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE"}
    if required_closed_cols.issubset(bur.columns):
        closed = bur[
            bur["CREDIT_ACTIVE"].eq("Closed")
            & bur["DAYS_ENDDATE_FACT"].notna()
            & bur["DAYS_CREDIT"].notna()
            & bur["DAYS_CREDIT_ENDDATE"].notna()
        ].copy()

        if not closed.empty:
            closed["bureau_credit_duration"] = closed["DAYS_ENDDATE_FACT"] - closed["DAYS_CREDIT"]
            closed["bureau_days_overrun"] = closed["DAYS_ENDDATE_FACT"] - closed["DAYS_CREDIT_ENDDATE"]

            closed_agg = closed.groupby("SK_ID_CURR").agg(
                bureau_mean_credit_duration=("bureau_credit_duration", "mean"),
                bureau_mean_days_overrun=("bureau_days_overrun", "mean"),
                bureau_pct_paid_early=("bureau_days_overrun", lambda x: (x > 0).mean()),
            )
            features = features.join(closed_agg)

    features = (
        features
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index()
    )

    return features


# =============================================================================
# bureau_balance: primero por SK_ID_BUREAU, luego por SK_ID_CURR usando bureau
# =============================================================================

def build_bureau_balance_features(bureau_balance: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Features mensuales de bureau_balance.

    bureau_balance no tiene SK_ID_CURR directamente.
    Por eso:
        1. Se agrega por SK_ID_BUREAU.
        2. Se une contra bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].
        3. Se vuelve a agregar por SK_ID_CURR.
    """
    bb = bureau_balance.copy()

    mora_states = ["1", "2", "3", "4", "5"]

    bb["bb_is_mora"] = bb["STATUS"].isin(mora_states).astype(int)
    bb["bb_is_0"] = bb["STATUS"].eq("0").astype(int)
    bb["bb_is_C"] = bb["STATUS"].eq("C").astype(int)
    bb["bb_is_X"] = bb["STATUS"].eq("X").astype(int)
    bb["bb_severity"] = bb["STATUS"].apply(lambda x: int(x) if str(x).isdigit() else 0)

    by_bureau = (
        bb.groupby("SK_ID_BUREAU")
        .agg(
            bb_months_count=("MONTHS_BALANCE", "count"),
            bb_months_span=("MONTHS_BALANCE", lambda x: x.max() - x.min()),
            bb_n_mora=("bb_is_mora", "sum"),
            bb_n_status_0=("bb_is_0", "sum"),
            bb_n_status_C=("bb_is_C", "sum"),
            bb_n_status_X=("bb_is_X", "sum"),
            bb_max_severity=("bb_severity", "max"),
        )
        .reset_index()
    )

    by_bureau["bb_pct_mora"] = safe_div(by_bureau["bb_n_mora"], by_bureau["bb_months_count"])
    by_bureau["bb_pct_X"] = safe_div(by_bureau["bb_n_status_X"], by_bureau["bb_months_count"])
    by_bureau["bb_has_mora"] = (by_bureau["bb_n_mora"] > 0).astype(int)

    bureau_key = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates("SK_ID_BUREAU")
    by_curr = by_bureau.merge(bureau_key, on="SK_ID_BUREAU", how="inner")

    features = (
        by_curr.groupby("SK_ID_CURR")
        .agg(
            bb_credit_count=("SK_ID_BUREAU", "nunique"),
            bb_months_count_sum=("bb_months_count", "sum"),
            bb_months_count_mean=("bb_months_count", "mean"),
            bb_months_span_mean=("bb_months_span", "mean"),
            bb_months_span_max=("bb_months_span", "max"),
            bb_n_mora_sum=("bb_n_mora", "sum"),
            bb_has_mora_rate=("bb_has_mora", "mean"),
            bb_pct_mora_mean=("bb_pct_mora", "mean"),
            bb_pct_mora_max=("bb_pct_mora", "max"),
            bb_pct_X_mean=("bb_pct_X", "mean"),
            bb_max_severity_max=("bb_max_severity", "max"),
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index()
    )

    return features


# =============================================================================
# previous_application: features por SK_ID_CURR
# =============================================================================

def clean_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Reglas iniciales del notebook previous_application:
    - Conservar último registro válido por contrato previo.
    - Eliminar columnas con >=90% de nulos.
    - Eliminar aprobados con AMT_CREDIT o AMT_ANNUITY nulos o <=0.
    """
    prev = prev.copy()

    if "FLAG_LAST_APPL_PER_CONTRACT" in prev.columns:
        prev = (
            prev[prev["FLAG_LAST_APPL_PER_CONTRACT"].eq("Y")]
            .drop(columns=["FLAG_LAST_APPL_PER_CONTRACT"])
        )

    null_ratio = prev.isna().mean()
    cols_to_drop = null_ratio[null_ratio >= 0.90].index.tolist()
    prev = prev.drop(columns=cols_to_drop, errors="ignore")

    needed = {"NAME_CONTRACT_STATUS", "AMT_CREDIT", "AMT_ANNUITY"}
    if needed.issubset(prev.columns):
        mask_bad_approved = (
            prev["NAME_CONTRACT_STATUS"].eq("Approved")
            & (
                prev["AMT_CREDIT"].isna()
                | prev["AMT_CREDIT"].le(0)
                | prev["AMT_ANNUITY"].isna()
                | prev["AMT_ANNUITY"].le(0)
            )
        )
        prev = prev.loc[~mask_bad_approved].copy()

    return prev


def build_previous_application_features(previous_application: pd.DataFrame) -> pd.DataFrame:
    """
    Features de solicitudes previas agregadas por SK_ID_CURR.
    Replica la función del notebook de previous_application.
    """
    prev = clean_previous_application(previous_application)

    id_col = "SK_ID_CURR"
    status_col = "NAME_CONTRACT_STATUS"

    features = prev.groupby(id_col).size().to_frame("prev_app_count")

    # Estados de solicitudes previas.
    features["prev_approved_rate"] = prev[status_col].eq("Approved").groupby(prev[id_col]).mean()
    features["prev_refused_rate"] = prev[status_col].eq("Refused").groupby(prev[id_col]).mean()
    features["prev_canceled_rate"] = prev[status_col].eq("Canceled").groupby(prev[id_col]).mean()
    features["prev_unused_offer_rate"] = prev[status_col].eq("Unused offer").groupby(prev[id_col]).mean()

    # Tipo de contrato.
    contract_type = prev["NAME_CONTRACT_TYPE"].astype(str).str.lower()
    features["prev_contract_type_cash_rate"] = contract_type.str.contains("cash", na=False).groupby(prev[id_col]).mean()
    features["prev_contract_type_consumer_rate"] = contract_type.str.contains("consumer", na=False).groupby(prev[id_col]).mean()
    features["prev_contract_type_revolving_rate"] = contract_type.str.contains("revolving", na=False).groupby(prev[id_col]).mean()

    # Anualidad aprobada.
    mask_annuity = prev[status_col].eq("Approved") & prev["AMT_ANNUITY"].gt(0)
    annuity_features = (
        prev.loc[mask_annuity]
        .groupby(id_col)["AMT_ANNUITY"]
        .agg(prev_annuity_approved_mean="mean", prev_annuity_approved_max="max")
    )
    features = features.join(annuity_features)

    # Monto solicitado en aprobados y rechazados.
    mask_application = prev[status_col].isin(["Approved", "Refused"]) & prev["AMT_APPLICATION"].gt(0)
    application_features = (
        prev.loc[mask_application]
        .groupby(id_col)["AMT_APPLICATION"]
        .agg(prev_application_mean="mean", prev_application_max="max")
    )
    features = features.join(application_features)

    # Monto de crédito aprobado.
    mask_credit = prev[status_col].eq("Approved") & prev["AMT_CREDIT"].gt(0)
    credit_features = (
        prev.loc[mask_credit]
        .groupby(id_col)["AMT_CREDIT"]
        .agg(prev_credit_approved_mean="mean")
    )
    features = features.join(credit_features)

    # Ratio crédito / solicitud en aprobados válidos.
    mask_ratio = (
        prev[status_col].eq("Approved")
        & prev["AMT_CREDIT"].gt(0)
        & prev["AMT_APPLICATION"].gt(0)
    )
    ratio_df = prev.loc[mask_ratio, [id_col, "AMT_CREDIT", "AMT_APPLICATION"]].copy()
    ratio_df["credit_application_ratio"] = ratio_df["AMT_CREDIT"] / ratio_df["AMT_APPLICATION"]
    ratio_features = (
        ratio_df.groupby(id_col)["credit_application_ratio"]
        .mean()
        .to_frame("prev_credit_application_ratio_mean")
    )
    features = features.join(ratio_features)

    # Precio de bienes.
    mask_goods = prev["AMT_GOODS_PRICE"].gt(0)
    goods_features = (
        prev.loc[mask_goods]
        .groupby(id_col)["AMT_GOODS_PRICE"]
        .agg(prev_goods_price_mean="mean", prev_goods_price_max="max")
    )
    features = features.join(goods_features)

    # Plazo.
    mask_cnt_payment = prev["CNT_PAYMENT"].gt(0)
    cnt_payment_features = (
        prev.loc[mask_cnt_payment]
        .groupby(id_col)["CNT_PAYMENT"]
        .agg(prev_cnt_payment_mean="mean", prev_cnt_payment_max="max")
    )
    features = features.join(cnt_payment_features)

    # Motivos de rechazo dentro de rechazadas.
    refused_mask = prev[status_col].eq("Refused")
    refused_total = refused_mask.groupby(prev[id_col]).sum()
    reject_reason = prev["CODE_REJECT_REASON"].astype(str).str.upper()

    features["prev_reject_hc_rate"] = (
        (refused_mask & reject_reason.eq("HC"))
        .groupby(prev[id_col]).sum()
        .div(refused_total.replace(0, np.nan))
    )
    features["prev_reject_limit_rate"] = (
        (refused_mask & reject_reason.eq("LIMIT"))
        .groupby(prev[id_col]).sum()
        .div(refused_total.replace(0, np.nan))
    )
    features["prev_reject_sco_rate"] = (
        (refused_mask & reject_reason.eq("SCO"))
        .groupby(prev[id_col]).sum()
        .div(refused_total.replace(0, np.nan))
    )

    # Yield.
    yield_group = prev["NAME_YIELD_GROUP"].astype(str).str.lower()
    features["prev_yield_high_rate"] = yield_group.eq("high").groupby(prev[id_col]).mean()
    features["prev_yield_low_rate"] = yield_group.str.contains("low", na=False).groupby(prev[id_col]).mean()

    # Seguro sobre aprobados.
    approved_mask = prev[status_col].eq("Approved")
    approved_total = approved_mask.groupby(prev[id_col]).sum()
    insured = pd.to_numeric(prev["NFLAG_INSURED_ON_APPROVAL"], errors="coerce").fillna(0).eq(1)

    features["prev_insured_on_approval_count"] = (
        (approved_mask & insured)
        .groupby(prev[id_col])
        .sum()
    )
    features["prev_insured_on_approval_rate"] = (
        features["prev_insured_on_approval_count"]
        .div(approved_total.replace(0, np.nan))
    )

    final_cols = [
        "prev_app_count",
        "prev_approved_rate", "prev_refused_rate", "prev_canceled_rate", "prev_unused_offer_rate",
        "prev_contract_type_cash_rate", "prev_contract_type_consumer_rate", "prev_contract_type_revolving_rate",
        "prev_annuity_approved_mean", "prev_annuity_approved_max",
        "prev_application_mean", "prev_application_max",
        "prev_credit_approved_mean", "prev_credit_application_ratio_mean",
        "prev_goods_price_mean", "prev_goods_price_max",
        "prev_cnt_payment_mean", "prev_cnt_payment_max",
        "prev_reject_hc_rate", "prev_reject_limit_rate", "prev_reject_sco_rate",
        "prev_yield_high_rate", "prev_yield_low_rate",
        "prev_insured_on_approval_rate", "prev_insured_on_approval_count",
    ]

    features = (
        features
        .reindex(columns=final_cols)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index()
    )

    return features


# =============================================================================
# installments_payments: features por SK_ID_CURR
# =============================================================================

def build_installments_payment_features(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Features de comportamiento de pago de cuotas.
    Replica el notebook installments_payments.
    """
    inst = installments.copy()

    id_col = "SK_ID_CURR"
    prev_id_col = "SK_ID_PREV"

    inst["inst_payment_delay_days"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]

    inst["inst_is_missing_payment"] = (
        inst["DAYS_ENTRY_PAYMENT"].isna() | inst["AMT_PAYMENT"].isna()
    ).astype(int)

    inst["inst_is_paid_observed"] = (
        inst["DAYS_ENTRY_PAYMENT"].notna() & inst["AMT_PAYMENT"].notna()
    ).astype(int)

    inst["inst_is_late_payment"] = (
        inst["inst_is_paid_observed"].eq(1) & inst["inst_payment_delay_days"].gt(0)
    ).astype(int)

    inst["inst_is_on_time_payment"] = (
        inst["inst_is_paid_observed"].eq(1) & inst["inst_payment_delay_days"].le(0)
    ).astype(int)

    inst["inst_days_past_due"] = inst["inst_payment_delay_days"].clip(lower=0)

    inst["inst_days_past_due_positive"] = np.where(
        inst["inst_is_late_payment"].eq(1),
        inst["inst_payment_delay_days"],
        np.nan,
    )

    inst["inst_payment_ratio"] = np.where(
        inst["AMT_INSTALMENT"].gt(0) & inst["AMT_PAYMENT"].notna(),
        inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"],
        np.nan,
    )
    inst["inst_payment_ratio"] = inst["inst_payment_ratio"].replace([np.inf, -np.inf], np.nan)

    inst["inst_is_underpayment"] = (
        inst["AMT_PAYMENT"].notna()
        & inst["AMT_INSTALMENT"].notna()
        & inst["AMT_INSTALMENT"].gt(0)
        & inst["AMT_PAYMENT"].lt(inst["AMT_INSTALMENT"])
    ).astype(int)

    inst["inst_is_overpayment"] = (
        inst["AMT_PAYMENT"].notna()
        & inst["AMT_INSTALMENT"].notna()
        & inst["AMT_INSTALMENT"].gt(0)
        & inst["AMT_PAYMENT"].gt(inst["AMT_INSTALMENT"])
    ).astype(int)

    inst["inst_is_credit_card"] = inst["NUM_INSTALMENT_VERSION"].eq(0).astype(int)

    features = (
        inst.groupby(id_col)
        .agg(
            inst_prev_credit_count=(prev_id_col, "nunique"),
            inst_total_installments_count=("NUM_INSTALMENT_NUMBER", "count"),
            inst_late_payment_rate=("inst_is_late_payment", "mean"),
            inst_days_past_due_mean=("inst_days_past_due_positive", "mean"),
            inst_days_past_due_max=("inst_days_past_due", "max"),
            inst_on_time_payment_rate=("inst_is_on_time_payment", "mean"),
            inst_missing_payment_rate=("inst_is_missing_payment", "mean"),
            inst_payment_ratio_mean=("inst_payment_ratio", "mean"),
            inst_underpayment_rate=("inst_is_underpayment", "mean"),
            inst_overpayment_rate=("inst_is_overpayment", "mean"),
            inst_amt_instalment_mean=("AMT_INSTALMENT", "mean"),
            inst_amt_payment_mean=("AMT_PAYMENT", "mean"),
            inst_calendar_version_max=("NUM_INSTALMENT_VERSION", "max"),
            inst_credit_card_rate=("inst_is_credit_card", "mean"),
        )
    )

    total_amounts = (
        inst.groupby(id_col)
        .agg(
            total_amt_payment=("AMT_PAYMENT", "sum"),
            total_amt_instalment=("AMT_INSTALMENT", "sum"),
        )
    )
    features["inst_total_payment_ratio"] = (
        total_amounts["total_amt_payment"] / total_amounts["total_amt_instalment"].replace(0, np.nan)
    )

    final_cols = [
        "inst_prev_credit_count", "inst_total_installments_count",
        "inst_late_payment_rate", "inst_days_past_due_mean", "inst_days_past_due_max",
        "inst_on_time_payment_rate", "inst_missing_payment_rate",
        "inst_payment_ratio_mean", "inst_underpayment_rate", "inst_overpayment_rate",
        "inst_total_payment_ratio", "inst_amt_instalment_mean", "inst_amt_payment_mean",
        "inst_calendar_version_max", "inst_credit_card_rate",
    ]

    features = (
        features
        .reindex(columns=final_cols)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index()
    )

    return features


# =============================================================================
# POS_CASH_balance: features por SK_ID_CURR
# =============================================================================

def build_pos_cash_features(pos_cash_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Features POS/CASH agregadas por SK_ID_CURR.
    Replica la función del notebook pos_cash_balance.
    """
    pos = pos_cash_balance.copy()

    id_col = "SK_ID_CURR"
    prev_id_col = "SK_ID_PREV"
    status_col = "NAME_CONTRACT_STATUS"

    pos["pos_is_active"] = pos[status_col].eq("Active").astype(int)
    pos["pos_is_completed"] = pos[status_col].eq("Completed").astype(int)
    pos["pos_has_dpd_def"] = pos["SK_DPD_DEF"].gt(0).astype(int)

    recent_idx = pos.groupby([id_col, prev_id_col])["MONTHS_BALANCE"].idxmax()
    recent_pos = pos.loc[recent_idx].copy()
    recent_pos["pos_recent_is_active"] = recent_pos[status_col].eq("Active").astype(int)

    features = (
        pos.groupby(id_col)
        .agg(
            pos_cash_active_rate=("pos_is_active", "mean"),
            pos_cash_completed_rate=("pos_is_completed", "mean"),
            pos_cash_dpd_def_positive_rate=("pos_has_dpd_def", "mean"),
            pos_cash_dpd_def_max=("SK_DPD_DEF", "max"),
        )
    )

    recent_features = (
        recent_pos.groupby(id_col)
        .agg(
            pos_cash_recent_active_count=("pos_recent_is_active", "sum"),
            pos_cash_recent_installments_future_mean=("CNT_INSTALMENT_FUTURE", "mean"),
            pos_cash_recent_installments_future_max=("CNT_INSTALMENT_FUTURE", "max"),
        )
    )

    features = (
        features
        .join(recent_features)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index()
    )

    return features


# =============================================================================
# credit_card_balance: features por SK_ID_CURR
# =============================================================================

def build_credit_card_features(credit_card_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Features de credit_card_balance.

    En el notebook se identificó que AMT_RECEIVABLE_PRINCIPAL y AMT_RECIVABLE
    eran redundantes frente a AMT_TOTAL_RECEIVABLE, por eso se eliminan.
    Como el notebook no tenía una función final de features, se agregan variables
    consistentes con el EDA: estados, mora, balances, pagos, límite y utilización.
    """
    cc = credit_card_balance.copy()
    id_col = "SK_ID_CURR"
    prev_id_col = "SK_ID_PREV"
    status_col = "NAME_CONTRACT_STATUS"

    cc = cc.drop(columns=["AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE"], errors="ignore")

    cc["cc_is_active"] = cc[status_col].eq("Active").astype(int)
    cc["cc_is_completed"] = cc[status_col].eq("Completed").astype(int)
    cc["cc_has_dpd"] = cc["SK_DPD"].gt(0).astype(int)
    cc["cc_has_dpd_def"] = cc["SK_DPD_DEF"].gt(0).astype(int)

    if {"AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(cc.columns):
        cc["cc_utilization"] = safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    else:
        cc["cc_utilization"] = np.nan

    if {"AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY"}.issubset(cc.columns):
        cc["cc_payment_min_ratio"] = safe_div(cc["AMT_PAYMENT_CURRENT"], cc["AMT_INST_MIN_REGULARITY"])
    else:
        cc["cc_payment_min_ratio"] = np.nan

    agg_dict = {
        "cc_months_count": ("MONTHS_BALANCE", "count"),
        "cc_prev_credit_count": (prev_id_col, "nunique"),
        "cc_active_rate": ("cc_is_active", "mean"),
        "cc_completed_rate": ("cc_is_completed", "mean"),
        "cc_dpd_positive_rate": ("cc_has_dpd", "mean"),
        "cc_dpd_def_positive_rate": ("cc_has_dpd_def", "mean"),
        "cc_dpd_max": ("SK_DPD", "max"),
        "cc_dpd_def_max": ("SK_DPD_DEF", "max"),
        "cc_utilization_mean": ("cc_utilization", "mean"),
        "cc_utilization_max": ("cc_utilization", "max"),
        "cc_payment_min_ratio_mean": ("cc_payment_min_ratio", "mean"),
    }

    optional_aggs = {
        "AMT_BALANCE": [("cc_amt_balance_mean", "mean"), ("cc_amt_balance_max", "max")],
        "AMT_CREDIT_LIMIT_ACTUAL": [("cc_credit_limit_mean", "mean"), ("cc_credit_limit_max", "max")],
        "AMT_TOTAL_RECEIVABLE": [("cc_total_receivable_mean", "mean"), ("cc_total_receivable_max", "max")],
        "AMT_DRAWINGS_CURRENT": [("cc_drawings_current_mean", "mean"), ("cc_drawings_current_max", "max")],
        "AMT_DRAWINGS_ATM_CURRENT": [("cc_drawings_atm_current_mean", "mean")],
        "AMT_DRAWINGS_POS_CURRENT": [("cc_drawings_pos_current_mean", "mean")],
        "AMT_PAYMENT_CURRENT": [("cc_payment_current_mean", "mean"), ("cc_payment_current_max", "max")],
        "AMT_INST_MIN_REGULARITY": [("cc_min_payment_mean", "mean")],
        "CNT_INSTALMENT_MATURE_CUM": [("cc_installments_mature_cum_max", "max")],
    }

    for source_col, named_aggs in optional_aggs.items():
        if source_col in cc.columns:
            for output_col, func in named_aggs:
                agg_dict[output_col] = (source_col, func)

    features = cc.groupby(id_col).agg(**agg_dict)

    # Snapshot más reciente por crédito previo.
    if {"MONTHS_BALANCE", id_col, prev_id_col}.issubset(cc.columns):
        recent_idx = cc.groupby([id_col, prev_id_col])["MONTHS_BALANCE"].idxmax()
        recent = cc.loc[recent_idx].copy()

        recent_aggs = {
            "cc_recent_active_count": ("cc_is_active", "sum"),
            "cc_recent_utilization_mean": ("cc_utilization", "mean"),
            "cc_recent_utilization_max": ("cc_utilization", "max"),
        }
        if "AMT_BALANCE" in recent.columns:
            recent_aggs["cc_recent_balance_mean"] = ("AMT_BALANCE", "mean")
        if "AMT_CREDIT_LIMIT_ACTUAL" in recent.columns:
            recent_aggs["cc_recent_credit_limit_mean"] = ("AMT_CREDIT_LIMIT_ACTUAL", "mean")

        recent_features = recent.groupby(id_col).agg(**recent_aggs)
        features = features.join(recent_features)

    features = (
        features
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index()
    )

    return features


# =============================================================================
# Dataset final
# =============================================================================

def build_final_train_dataset(data_dir: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Construye el dataset final:
        application_train LEFT JOIN features históricas agregadas.
    """
    print(f"[INFO] Carpeta data: {data_dir}")

    app = read_table(data_dir, "application_train", required=True)
    app = clean_application_train(app)

    print(f"[INFO] Base application_train limpia: {app.shape[0]:,} filas x {app.shape[1]:,} columnas")

    # Cargar tablas históricas.
    bureau = read_table(data_dir, "bureau", required=False)
    bureau_balance = read_table(data_dir, "bureau_balance", required=False)
    previous_application = read_table(data_dir, "previous_application", required=False)
    installments = read_table(data_dir, "installments_payments", required=False)
    pos_cash = read_table(data_dir, "pos_cash_balance", required=False)
    credit_card = read_table(data_dir, "credit_card_balance", required=False)

    final = app.copy()
    initial_rows = len(final)

    if bureau is not None:
        bureau_features = build_bureau_features(bureau)
        final = left_join_features(final, bureau_features, "SK_ID_CURR", "bureau", "no_bureau_history")

    if bureau is not None and bureau_balance is not None:
        bb_features = build_bureau_balance_features(bureau_balance, bureau)
        final = left_join_features(final, bb_features, "SK_ID_CURR", "bureau_balance", "no_bureau_balance_history")

    if previous_application is not None:
        prev_features = build_previous_application_features(previous_application)
        final = left_join_features(final, prev_features, "SK_ID_CURR", "previous_application", "no_previous_application_history")

    if installments is not None:
        inst_features = build_installments_payment_features(installments)
        final = left_join_features(final, inst_features, "SK_ID_CURR", "installments_payments", "no_installments_history")

    if pos_cash is not None:
        pos_features = build_pos_cash_features(pos_cash)
        final = left_join_features(final, pos_features, "SK_ID_CURR", "pos_cash_balance", "no_pos_cash_history")

    if credit_card is not None:
        cc_features = build_credit_card_features(credit_card)
        final = left_join_features(final, cc_features, "SK_ID_CURR", "credit_card_balance", "no_credit_card_history")

    # Validaciones finales.
    if len(final) != initial_rows:
        raise RuntimeError(f"El dataset final cambió filas: {initial_rows:,} -> {len(final):,}")

    final = ensure_unique_by_id(final, "SK_ID_CURR", "dataset_final")

    if len(final) != initial_rows:
        raise RuntimeError(
            "Después de asegurar unicidad, el número de filas no coincide con application_train. "
            "Revisa duplicados en la base."
        )

    print("=" * 80)
    print("[OK] Dataset final construido")
    print(f"Filas finales    : {final.shape[0]:,}")
    print(f"Columnas finales : {final.shape[1]:,}")
    print(f"SK_ID_CURR únicos: {final['SK_ID_CURR'].nunique():,}")
    print("=" * 80)

    if output_path is None:
        output_path = data_dir / "final_train_features.parquet"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output_path, index=False)
    print(f"[OK] Archivo guardado en: {output_path}")

    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye un dataset final con application_train como tabla principal."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Ruta a la carpeta data/. Si no se pasa, se busca automáticamente.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del parquet final. Default: data/final_train_features.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir() if args.data_dir is None else args.data_dir
    build_final_train_dataset(data_dir=data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
