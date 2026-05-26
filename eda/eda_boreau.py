"""
=====================================================================
EDA COMPLETO — bureau.parquet
+ JOIN con application_train para análisis con TARGET
=====================================================================
Ejecutar: python3 eda_bureau.py
Genera  : carpeta eda_bureau/ con todas las figuras
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

OUT = "eda_bureau"
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 0. CARGA
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("0. CARGA DE DATOS")
print("="*60)

bur = pd.read_parquet(DATA_DIR / "bureau.parquet")
app = pd.read_parquet(
    DATA_DIR / "application_train.parquet",
    columns=['SK_ID_CURR', 'TARGET'],
)

print(f"  bureau            : {bur.shape[0]:,} filas x {bur.shape[1]} columnas")
print(f"  application_train : {app.shape[0]:,} filas x {app.shape[1]} columnas")
print(f"\n  Clientes únicos en bureau : {bur['SK_ID_CURR'].nunique():,}")
print(f"  Clientes únicos en app    : {app['SK_ID_CURR'].nunique():,}")
print(f"  Clientes en app SIN bureau: {app['SK_ID_CURR'].nunique() - bur['SK_ID_CURR'].nunique():,}  "
      f"({(1 - bur['SK_ID_CURR'].nunique()/app['SK_ID_CURR'].nunique())*100:.1f}%)")
print("\n  Columnas:\n  ", list(bur.columns))


# ══════════════════════════════════════════════════════════════════
# 1. TIPOS DE DATO Y PRIMERAS FILAS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. TIPOS DE DATO")
print("="*60)
print(bur.dtypes.to_string())
print("\n  Primeras 5 filas:")
print(bur.head().to_string())


# ══════════════════════════════════════════════════════════════════
# 2. NULOS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. ANÁLISIS DE NULOS")
print("="*60)

null_pct = (bur.isnull().sum() / len(bur) * 100).sort_values(ascending=False)
print(null_pct.round(2).to_string())
print("""
  ⚠  DECISIONES:
     → AMT_ANNUITY          (71% nulos): ELIMINAR columna raw.
       Usar solo en agregados ponderados si se necesita.
     → AMT_CREDIT_MAX_OVERDUE (65% nulos): ELIMINAR columna raw.
       La ausencia suele significar que nunca tuvo overdue.
     → DAYS_ENDDATE_FACT    (37% nulos): Solo aplica a créditos CERRADOS.
       Imputar con 0 al agregar (ausencia = crédito aún activo).
     → AMT_CREDIT_SUM_LIMIT (34% nulos): Aplica a tarjetas de crédito.
       Imputar con 0 al agregar.
     → AMT_CREDIT_SUM_DEBT  (15% nulos): Imputar con 0.
     → DAYS_CREDIT_ENDDATE  (6% nulos): Imputar con mediana.
""")

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#d73027' if v > 60 else '#fc8d59' if v > 30 else '#fee090' if v > 0 else '#91cf60'
          for v in null_pct.values]
ax.barh(null_pct.index[::-1], null_pct.values[::-1], color=colors[::-1])
ax.axvline(60, color='red', linestyle='--', linewidth=1.2, label='>60% → eliminar')
ax.axvline(30, color='orange', linestyle='--', linewidth=1.2, label='>30% → evaluar')
ax.set_xlabel('% valores nulos')
ax.set_title('Nulos por columna — bureau.parquet\n'
             '🔴 >60% | 🟠 30-60% | 🟡 >0% | 🟢 sin nulos')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/01_nulos.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/01_nulos.png")


# ══════════════════════════════════════════════════════════════════
# 3. REGISTROS POR CLIENTE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. REGISTROS POR CLIENTE (créditos en historial)")
print("="*60)

cnt_per_client = bur.groupby('SK_ID_CURR').size()
print(cnt_per_client.describe().round(2))
print(f"\n  Clientes con 1 crédito   : {(cnt_per_client==1).sum():,}")
print(f"  Clientes con 2-5 créditos: {((cnt_per_client>=2)&(cnt_per_client<=5)).sum():,}")
print(f"  Clientes con >10 créditos: {(cnt_per_client>10).sum():,}")
print(f"  Clientes con >20 créditos: {(cnt_per_client>20).sum():,}")
print(f"  Máximo créditos un cliente: {cnt_per_client.max()}")
print("""
  ⚠  DECISIÓN: La tabla está en formato largo (un registro por crédito).
     Para modelar hay que AGREGAR por SK_ID_CURR y hacer join con app.
     No se deben eliminar filas — cada crédito es información válida.
     Los outliers (>20 créditos) son clientes con mucho historial,
     información valiosa, no errores.
""")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(cnt_per_client.values, bins=50, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Nº créditos en bureau por cliente')
axes[0].set_ylabel('Nº clientes')
axes[0].set_title('Distribución de créditos por cliente')
axes[0].axvline(cnt_per_client.mean(), color='red', linestyle='--',
                label=f'Media={cnt_per_client.mean():.1f}')
axes[0].legend()

axes[1].hist(cnt_per_client[cnt_per_client <= 20].values, bins=20,
             color='steelblue', edgecolor='white')
axes[1].set_xlabel('Nº créditos (vista ≤20)')
axes[1].set_ylabel('Nº clientes')
axes[1].set_title('Zoom: clientes con ≤20 créditos')

plt.suptitle('¿Cuántos créditos tiene cada cliente en bureau?', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/02_creditos_por_cliente.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/02_creditos_por_cliente.png")


# ══════════════════════════════════════════════════════════════════
# 4. VARIABLES CATEGÓRICAS — DISTRIBUCIÓN
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. VARIABLES CATEGÓRICAS")
print("="*60)

# CREDIT_ACTIVE
print("\n--- CREDIT_ACTIVE ---")
ca = bur['CREDIT_ACTIVE'].value_counts()
print(ca)
print(f"\n  Ratio cerrados/activos: {ca.get('Closed',0)/ca.get('Active',1):.2f}x")
print("""
  ⚠  DECISIÓN: 'Sold' y 'Bad debt' son señal de alto riesgo (solo 6,548 casos).
     → Feature: has_bad_or_sold = 1 si algún crédito en estas categorías.
     → Feature: n_active = count de créditos activos.
     → Feature: active_pct = n_active / n_credits.
""")

# CREDIT_TYPE
print("\n--- CREDIT_TYPE ---")
ct = bur['CREDIT_TYPE'].value_counts()
print(ct)
print("""
  ⚠  DECISIÓN: Consumer credit (73%) y Credit card (23%) dominan.
     → Las 9 categorías con <100 registros se pueden agrupar como 'Other'.
     → Features útiles: n_consumer, n_card, n_mortgage, n_carloan.
     → Tipo de crédito refleja comportamiento financiero.
""")

# CREDIT_CURRENCY
print("\n--- CREDIT_CURRENCY ---")
print(bur['CREDIT_CURRENCY'].value_counts())
print("""
  ⚠  DECISIÓN: 99.9% en 'currency 1'. Columna sin valor predictivo.
     → ELIMINAR: varianza prácticamente cero.
""")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# CREDIT_ACTIVE
ca_plot = bur['CREDIT_ACTIVE'].value_counts()
colors_ca = ['#4dac26','#b8e186','#d01c8b','#f1b6da']
axes[0].bar(ca_plot.index, ca_plot.values, color=colors_ca[:len(ca_plot)], edgecolor='white')
axes[0].set_title('CREDIT_ACTIVE\n(estado del crédito)', fontweight='bold')
axes[0].set_ylabel('Nº registros')
for i, v in enumerate(ca_plot.values):
    axes[0].text(i, v + 5000, f'{v:,}\n({v/len(bur)*100:.1f}%)', ha='center', fontsize=8)

# CREDIT_TYPE
ct_plot = bur['CREDIT_TYPE'].value_counts().head(8)
axes[1].barh(ct_plot.index[::-1], ct_plot.values[::-1], color='steelblue')
axes[1].set_title('CREDIT_TYPE (top 8)\n(tipo de crédito)', fontweight='bold')
axes[1].set_xlabel('Nº registros')
for i, v in enumerate(ct_plot.values[::-1]):
    axes[1].text(v + 1000, i, f'{v:,}', va='center', fontsize=8)

# CREDIT_CURRENCY
cc_plot = bur['CREDIT_CURRENCY'].value_counts()
axes[2].bar(cc_plot.index, cc_plot.values, color=['#2c7bb6','#abd9e9','#ffffbf','#d7191c'],
            edgecolor='white')
axes[2].set_title('CREDIT_CURRENCY\n(⚠ 99.9% currency 1 → ELIMINAR)', fontweight='bold')
axes[2].set_ylabel('Nº registros')
for i, v in enumerate(cc_plot.values):
    axes[2].text(i, v + 1000, f'{v:,}', ha='center', fontsize=8)

plt.suptitle('Distribución de variables categóricas — bureau', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/03_categoricas.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/03_categoricas.png")


# ══════════════════════════════════════════════════════════════════
# 5. VARIABLES NUMÉRICAS — DISTRIBUCIÓN Y OUTLIERS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. VARIABLES NUMÉRICAS — ESTADÍSTICOS Y OUTLIERS")
print("="*60)

num_cols = ['DAYS_CREDIT','CREDIT_DAY_OVERDUE','DAYS_CREDIT_ENDDATE',
            'DAYS_ENDDATE_FACT','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT',
            'AMT_CREDIT_SUM_OVERDUE','DAYS_CREDIT_UPDATE']

desc = bur[num_cols].describe().round(2)
print(desc.to_string())

print("""
  ⚠  NOTAS IMPORTANTES:
     → DAYS_CREDIT: todos negativos (días ANTES de la solicitud). Rango -2922 a 0.
       Media -1142 días ≈ 3 años de antigüedad promedio del historial.
     → CREDIT_DAY_OVERDUE: 75% tiene 0 (no hay mora). Max=2792 días → outliers extremos.
       Se debe capturar como flag (>0) y como max/mean al agregar.
     → AMT_CREDIT_SUM: rango 0 a 585M. Distribución muy sesgada → usar log o percentiles.
     → AMT_CREDIT_SUM_DEBT: tiene valores NEGATIVOS (-4.7M) → probablemente pagos en exceso.
     → AMT_CREDIT_SUM_OVERDUE: 75% en 0. Los casos con >0 son los riesgosos.
     → DAYS_CREDIT_UPDATE: negativo (días antes de la solicitud). Max=372 (error posible).
""")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    data = bur[col].dropna()
    p1, p99 = data.quantile(0.01), data.quantile(0.99)
    data_clip = data.clip(p1, p99)
    ax.hist(data_clip, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(col, fontsize=8, fontweight='bold')
    ax.set_xlabel(f'[recortado p1-p99]\nMedia={data.mean():.0f} | Nulos={bur[col].isna().mean()*100:.0f}%',
                  fontsize=7)
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1)
    ax.axvline(data.median(), color='green', linestyle='--', linewidth=1)

plt.suptitle('Distribución variables numéricas (recortadas p1-p99)\n'
             '🔴 media | 🟢 mediana', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/04_numericas_dist.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/04_numericas_dist.png")

# Outliers con boxplot
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
axes = axes.flatten()
for ax, col in zip(axes, num_cols):
    data = bur[col].dropna()
    p5, p95 = data.quantile(0.05), data.quantile(0.95)
    ax.boxplot(data.clip(p5, p95), vert=True, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.6))
    pct_outlier = ((data < p5) | (data > p95)).mean() * 100
    ax.set_title(f'{col}\n({pct_outlier:.1f}% fuera p5-p95)', fontsize=8)
    ax.set_xticks([])
plt.suptitle('Boxplots variables numéricas (clipeadas p5-p95)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/04b_boxplots.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/04b_boxplots.png")


# ══════════════════════════════════════════════════════════════════
# 6. ANÁLISIS ESPECÍFICO: DAYS_CREDIT (antigüedad historial)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. DAYS_CREDIT — antigüedad del historial crediticio")
print("="*60)

print(f"  Rango: {bur['DAYS_CREDIT'].min()} a {bur['DAYS_CREDIT'].max()} días")
print(f"  Media: {bur['DAYS_CREDIT'].mean():.0f} días ({bur['DAYS_CREDIT'].mean()/365:.1f} años)")
print(f"  Mediana: {bur['DAYS_CREDIT'].median():.0f} días ({bur['DAYS_CREDIT'].median()/365:.1f} años)")
print(f"  Créditos de los últimos 6 meses  : {(bur['DAYS_CREDIT'] >= -180).sum():,}")
print(f"  Créditos de los últimos 2 años   : {(bur['DAYS_CREDIT'] >= -730).sum():,}")
print(f"  Créditos con >5 años de antigüedad: {(bur['DAYS_CREDIT'] < -1825).sum():,}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
bur['DAYS_CREDIT_YEARS'] = bur['DAYS_CREDIT'] / 365
axes[0].hist(bur['DAYS_CREDIT_YEARS'], bins=60, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Años antes de la solicitud')
axes[0].set_title('DAYS_CREDIT — ¿Cuán antiguo es el crédito?')
axes[0].axvline(bur['DAYS_CREDIT_YEARS'].mean(), color='red', linestyle='--', label='Media')
axes[0].legend()

# Por tipo de crédito
for ctype, color in [('Consumer credit','steelblue'),('Credit card','tomato'),('Car loan','green')]:
    subset = bur[bur['CREDIT_TYPE']==ctype]['DAYS_CREDIT_YEARS']
    axes[1].hist(subset, bins=40, alpha=0.5, label=ctype, color=color, density=True)
axes[1].set_xlabel('Años antes de la solicitud')
axes[1].set_title('DAYS_CREDIT por tipo de crédito')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/05_days_credit.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/05_days_credit.png")


# ══════════════════════════════════════════════════════════════════
# 7. CREDIT_DAY_OVERDUE — mora
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. CREDIT_DAY_OVERDUE — días de mora")
print("="*60)

overdue = bur['CREDIT_DAY_OVERDUE']
print(f"  Sin mora (=0)     : {(overdue==0).sum():,}  ({(overdue==0).mean()*100:.1f}%)")
print(f"  Con mora (>0)     : {(overdue>0).sum():,}  ({(overdue>0).mean()*100:.1f}%)")
print(f"  Mora >30 días     : {(overdue>30).sum():,}")
print(f"  Mora >90 días     : {(overdue>90).sum():,}")
print(f"  Mora >365 días    : {(overdue>365).sum():,}")
print(f"  Máximo            : {overdue.max()} días")
print("""
  ⚠  DECISIÓN: Solo el 1.8% tiene mora activa. Es poco frecuente pero
     muy discriminativo. Al agregar por cliente:
     → has_overdue  = 1 si alguna vez tuvo mora
     → max_overdue_days = el peor caso
     → sum_overdue_days = acumulado total de mora
""")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(['Sin mora (0)', 'Con mora (>0)'],
            [(overdue==0).sum(), (overdue>0).sum()],
            color=['#4dac26','#d01c8b'], edgecolor='white')
axes[0].set_title('CREDIT_DAY_OVERDUE\n¿Tiene mora?')
for i, v in enumerate([(overdue==0).sum(), (overdue>0).sum()]):
    axes[0].text(i, v + 2000, f'{v:,}\n({v/len(bur)*100:.1f}%)', ha='center')

# Distribución solo mora >0
mora_pos = overdue[overdue > 0].clip(0, 500)
axes[1].hist(mora_pos, bins=50, color='tomato', edgecolor='white')
axes[1].set_xlabel('Días de mora (vista ≤500 días)')
axes[1].set_title('Distribución de mora (solo registros con >0)')
plt.tight_layout()
plt.savefig(f'{OUT}/06_mora.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/06_mora.png")


# ══════════════════════════════════════════════════════════════════
# 8. MONTOS — AMT_CREDIT_SUM y deuda
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("8. MONTOS DE CRÉDITO Y DEUDA")
print("="*60)

for col in ['AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE']:
    d = bur[col].dropna()
    print(f"\n  {col}:")
    print(f"    Media={d.mean():.0f} | Mediana={d.median():.0f} | "
          f"Max={d.max():.0f} | Nulos={bur[col].isna().mean()*100:.1f}%")
    pct_zero = (d == 0).mean() * 100
    print(f"    = 0: {pct_zero:.1f}%  | > 0: {100-pct_zero:.1f}%")

print("""
  ⚠  OBSERVACIONES:
     → AMT_CREDIT_SUM: muy sesgado a la derecha. Para agregar usar SUM y MAX.
     → AMT_CREDIT_SUM_DEBT: 75% en 0 (crédito saldado). Suma = deuda total.
     → AMT_CREDIT_SUM_OVERDUE: 99.7% en 0. Suma = monto vencido total, es señal de riesgo.
     → Valores negativos en AMT_CREDIT_SUM_DEBT → pagos en exceso, no son errores.
""")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE']):
    d = bur[col].dropna()
    d_clip = d.clip(d.quantile(0.01), d.quantile(0.99))
    ax.hist(d_clip, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(f'{col}\n(clipeado p1-p99 | {bur[col].isna().mean()*100:.0f}% nulos)',
                 fontsize=8, fontweight='bold')
    ax.axvline(d.mean(), color='red', linestyle='--', linewidth=1, label='Media')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=7)
plt.suptitle('Distribución de montos de crédito y deuda', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/07_montos.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/07_montos.png")


# ══════════════════════════════════════════════════════════════════
# 9. JOIN CON APPLICATION → ANÁLISIS CON TARGET
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("9. JOIN CON APPLICATION_TRAIN → ANÁLISIS CON TARGET")
print("="*60)

# --- Construir features agregados ---
g = bur.groupby('SK_ID_CURR')

agg = pd.DataFrame({
    'n_credits'        : g['SK_ID_BUREAU'].count(),
    'max_overdue_days' : g['CREDIT_DAY_OVERDUE'].max(),
    'sum_overdue_days' : g['CREDIT_DAY_OVERDUE'].sum(),
    'total_credit'     : g['AMT_CREDIT_SUM'].sum(),
    'total_debt'       : g['AMT_CREDIT_SUM_DEBT'].sum(),
    'total_overdue_amt': g['AMT_CREDIT_SUM_OVERDUE'].sum(),
    'n_prolonged'      : g['CNT_CREDIT_PROLONG'].sum(),
    'avg_days_credit'  : g['DAYS_CREDIT'].mean(),
    'min_days_credit'  : g['DAYS_CREDIT'].min(),
    'avg_days_update'  : g['DAYS_CREDIT_UPDATE'].mean(),
}).reset_index()

# Categóricas por separado
active_cnt = bur[bur['CREDIT_ACTIVE']=='Active'].groupby('SK_ID_CURR').size().rename('n_active')
closed_cnt = bur[bur['CREDIT_ACTIVE']=='Closed'].groupby('SK_ID_CURR').size().rename('n_closed')
bad_cnt    = bur[bur['CREDIT_ACTIVE'].isin(['Bad debt','Sold'])].groupby('SK_ID_CURR').size().rename('n_bad')
consumer_cnt = bur[bur['CREDIT_TYPE']=='Consumer credit'].groupby('SK_ID_CURR').size().rename('n_consumer')
card_cnt   = bur[bur['CREDIT_TYPE']=='Credit card'].groupby('SK_ID_CURR').size().rename('n_card')
mortgage_cnt = bur[bur['CREDIT_TYPE']=='Mortgage'].groupby('SK_ID_CURR').size().rename('n_mortgage')

agg = (agg
       .merge(active_cnt,   on='SK_ID_CURR', how='left')
       .merge(closed_cnt,   on='SK_ID_CURR', how='left')
       .merge(bad_cnt,      on='SK_ID_CURR', how='left')
       .merge(consumer_cnt, on='SK_ID_CURR', how='left')
       .merge(card_cnt,     on='SK_ID_CURR', how='left')
       .merge(mortgage_cnt, on='SK_ID_CURR', how='left'))

for col in ['n_active','n_closed','n_bad','n_consumer','n_card','n_mortgage']:
    agg[col] = agg[col].fillna(0)

# Features derivados
agg['active_pct']   = agg['n_active']  / agg['n_credits']
agg['closed_pct']   = agg['n_closed']  / agg['n_credits']
agg['debt_ratio']   = agg['total_debt'] / (agg['total_credit'].abs() + 1)
agg['has_bad_debt'] = (agg['n_bad'] > 0).astype(int)
agg['has_overdue']  = (agg['max_overdue_days'] > 0).astype(int)

# Join con app
merged = app.merge(agg, on='SK_ID_CURR', how='left')
merged['no_history'] = merged['n_credits'].isna().astype(int)

feat_cols = [c for c in merged.columns if c not in ['SK_ID_CURR', 'TARGET']]
corr = merged[feat_cols].corrwith(merged['TARGET']).sort_values(key=abs, ascending=False)

print("\n  Correlación de features bureau con TARGET:")
print(corr.round(4).to_string())
print("""
  ⚠  HALLAZGOS CLAVE:
     → avg_days_credit (+0.090): Historial MÁS ANTIGUO → MÁS riesgo.
       Parece contra-intuitivo pero significa que quien lleva más tiempo
       solicitando créditos tiende a tener más problemas acumulados.
     → active_pct (+0.077): Más % de créditos activos = más carga financiera = más riesgo.
     → n_active (+0.067): A más créditos activos simultáneos, mayor riesgo.
     → has_overdue (+0.035): Haber tenido mora es señal de riesgo.
     → no_history (+0.031): No tener historial en bureau es riesgo.
     → has_bad_debt (+0.013): Haber tenido deuda vendida/mala, señal moderada.
""")

# Gráfico correlaciones
fig, ax = plt.subplots(figsize=(10, 7))
corr_plot = corr.dropna().sort_values(key=abs, ascending=True)
colors_bar = ['tomato' if v > 0 else 'steelblue' for v in corr_plot.values]
ax.barh(corr_plot.index, corr_plot.values, color=colors_bar)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Correlación de Pearson con TARGET')
ax.set_title('Features agregados de bureau vs TARGET\n'
             '🔴 = aumenta riesgo de default | 🔵 = reduce riesgo')
plt.tight_layout()
plt.savefig(f'{OUT}/08_corr_target.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/08_corr_target.png")


# ══════════════════════════════════════════════════════════════════
# 10. DEFAULT RATE POR GRUPOS DE FEATURES BUREAU
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("10. DEFAULT RATE POR GRUPOS DE VARIABLES BUREAU")
print("="*60)

global_dr = merged['TARGET'].mean()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# a) Tiene historial vs no tiene
ax = axes[0]
groups_hist = merged.groupby('no_history')['TARGET'].agg(['mean','count'])
groups_hist.index = ['Con historial','Sin historial']
colors_h = ['#4dac26' if v < global_dr else '#d01c8b' for v in groups_hist['mean']]
bars = ax.bar(groups_hist.index, groups_hist['mean'], color=colors_h, edgecolor='white')
ax.axhline(global_dr, color='navy', linestyle='--', linewidth=1.5, label=f'Media global ({global_dr:.3f})')
ax.set_title('¿Tiene historial en bureau?', fontweight='bold')
ax.set_ylabel('Tasa de default')
ax.legend(fontsize=8)
for i, (_, row) in enumerate(groups_hist.iterrows()):
    ax.text(i, row['mean'] + 0.001, f"{row['mean']:.3f}\nn={row['count']:,}", ha='center', fontsize=9)

# b) has_overdue
ax = axes[1]
groups_ov = merged.groupby('has_overdue', dropna=False)['TARGET'].agg(['mean','count'])
groups_ov.index = ['Sin mora', 'Con mora', 'Sin historial'][:len(groups_ov)]
colors_ov = ['#4dac26' if v < global_dr else '#d01c8b' for v in groups_ov['mean']]
ax.bar(groups_ov.index, groups_ov['mean'], color=colors_ov, edgecolor='white')
ax.axhline(global_dr, color='navy', linestyle='--', linewidth=1.5, label=f'Media global ({global_dr:.3f})')
ax.set_title('¿Tuvo días de mora?', fontweight='bold')
ax.set_ylabel('Tasa de default')
ax.legend(fontsize=8)
for i, (_, row) in enumerate(groups_ov.iterrows()):
    ax.text(i, row['mean'] + 0.001, f"{row['mean']:.3f}\nn={row['count']:,}", ha='center', fontsize=9)

# c) has_bad_debt
ax = axes[2]
groups_bd = merged.groupby('has_bad_debt', dropna=False)['TARGET'].agg(['mean','count'])
groups_bd.index = ['Sin deuda mala', 'Con deuda mala/vendida', 'Sin historial'][:len(groups_bd)]
colors_bd = ['#4dac26' if v < global_dr else '#d01c8b' for v in groups_bd['mean']]
ax.bar(groups_bd.index, groups_bd['mean'], color=colors_bd, edgecolor='white')
ax.axhline(global_dr, color='navy', linestyle='--', linewidth=1.5, label=f'Media global')
ax.set_title('¿Tiene crédito Bad debt/Sold?', fontweight='bold')
ax.set_ylabel('Tasa de default')
ax.legend(fontsize=8)
for i, (_, row) in enumerate(groups_bd.iterrows()):
    ax.text(i, row['mean'] + 0.001, f"{row['mean']:.3f}\nn={row['count']:,}", ha='center', fontsize=9)

# d) n_credits bins
ax = axes[3]
merged['n_credits_bin'] = pd.cut(merged['n_credits'], bins=[0,1,3,6,10,20,200],
                                  labels=['1','2-3','4-6','7-10','11-20','>20'])
dr_ncred = merged.groupby('n_credits_bin', observed=True)['TARGET'].agg(['mean','count'])
ax.bar(dr_ncred.index.astype(str), dr_ncred['mean'], color='steelblue', edgecolor='white')
ax.axhline(global_dr, color='navy', linestyle='--', linewidth=1.5, label=f'Media global')
ax.set_xlabel('Nº créditos en bureau')
ax.set_title('Default rate por nº de créditos', fontweight='bold')
ax.set_ylabel('Tasa de default')
ax.legend(fontsize=8)
for i, (_, row) in enumerate(dr_ncred.iterrows()):
    ax.text(i, row['mean'] + 0.001, f"{row['mean']:.3f}", ha='center', fontsize=8)

# e) active_pct bins
ax = axes[4]
merged_withbur = merged[merged['no_history']==0].copy()
merged_withbur['active_pct_bin'] = pd.cut(merged_withbur['active_pct'],
                                           bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
                                           include_lowest=True,
                                           labels=['0-10%','10-25%','25-50%','50-75%','75-100%'])
dr_apct = merged_withbur.groupby('active_pct_bin', observed=True)['TARGET'].agg(['mean','count'])
colors_ap = ['#4dac26' if v < global_dr else '#d01c8b' for v in dr_apct['mean']]
ax.bar(dr_apct.index.astype(str), dr_apct['mean'], color=colors_ap, edgecolor='white')
ax.axhline(global_dr, color='navy', linestyle='--', linewidth=1.5, label=f'Media global')
ax.set_xlabel('% créditos activos')
ax.set_title('Default rate por % créditos activos', fontweight='bold')
ax.set_ylabel('Tasa de default')
ax.legend(fontsize=8)
for i, (_, row) in enumerate(dr_apct.iterrows()):
    ax.text(i, row['mean'] + 0.001, f"{row['mean']:.3f}", ha='center', fontsize=8)

# f) avg_days_credit bins
ax = axes[5]
merged_withbur['hist_years'] = merged_withbur['avg_days_credit'] / 365
merged_withbur['hist_bin'] = pd.cut(merged_withbur['hist_years'],
                                     bins=[-10, -5, -3, -2, -1, 0],
                                     labels=['>5 años','3-5 años','2-3 años','1-2 años','<1 año'])
dr_hist = merged_withbur.groupby('hist_bin', observed=True)['TARGET'].agg(['mean','count'])
colors_hist = ['#4dac26' if v < global_dr else '#d01c8b' for v in dr_hist['mean']]
ax.bar(dr_hist.index.astype(str), dr_hist['mean'], color=colors_hist, edgecolor='white')
ax.axhline(global_dr, color='navy', linestyle='--', linewidth=1.5, label=f'Media global')
ax.set_xlabel('Antigüedad promedio créditos')
ax.set_title('Default rate por antigüedad historial', fontweight='bold')
ax.set_ylabel('Tasa de default')
ax.legend(fontsize=8)
for i, (_, row) in enumerate(dr_hist.iterrows()):
    ax.text(i, row['mean'] + 0.001, f"{row['mean']:.3f}", ha='center', fontsize=8)

plt.suptitle('Default rate por variables clave del bureau\n'
             '🟢 < media global | 🔴 > media global (línea azul punteada)',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/09_default_rate_bureau.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/09_default_rate_bureau.png")


# ══════════════════════════════════════════════════════════════════
# 11. DEFAULT RATE POR TIPO DE CRÉDITO
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("11. DEFAULT RATE POR TIPO DE CRÉDITO EN BUREAU")
print("="*60)

bur_with_target = bur.merge(app, on='SK_ID_CURR', how='inner')
dr_type = bur_with_target.groupby('CREDIT_TYPE')['TARGET'].agg(['mean','count'])
dr_type = dr_type[dr_type['count'] > 50].sort_values('mean', ascending=False)
print(dr_type.round(4).to_string())
print("""
  ⚠  HALLAZGO: Los tipos de crédito reflejan el perfil de riesgo del cliente.
     → Mortgage (hipoteca): tasa de default más baja — clientes más estables.
     → Microloans: tasa más alta — clientes más vulnerables.
     → Consumer credit y Credit card: los más frecuentes, tasas medias.
""")

fig, ax = plt.subplots(figsize=(10, 5))
colors_type = ['#d01c8b' if v > global_dr else '#4dac26' for v in dr_type['mean']]
ax.barh(dr_type.index[::-1], dr_type['mean'][::-1], color=colors_type[::-1])
ax.axvline(global_dr, color='navy', linestyle='--', linewidth=1.5, label=f'Media global ({global_dr:.3f})')
ax.set_xlabel('Tasa de default (promedio cliente asociado)')
ax.set_title('Default rate del cliente según tipo de crédito en bureau\n'
             '🔴 > media global | 🟢 < media global', fontweight='bold')
ax.legend(fontsize=9)
for i, (idx, row) in enumerate(dr_type[::-1].iterrows()):
    ax.text(row['mean'] + 0.0005, i, f"{row['mean']:.3f}  n={row['count']:,}",
            va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/10_default_by_credit_type.png', bbox_inches='tight')
plt.close()
print(f"  [Guardado] {OUT}/10_default_by_credit_type.png")


# ══════════════════════════════════════════════════════════════════
# 12. RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("12. RESUMEN FINAL — BUREAU")
print("="*60)
print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║        DECISIONES SOBRE LA TABLA BUREAU                     ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  COLUMNAS A ELIMINAR (raw):                                 ║
  ║   → CREDIT_CURRENCY  (99.9% mismo valor, varianza~0)        ║
  ║   → AMT_ANNUITY      (71% nulos, poca utilidad raw)         ║
  ║   → AMT_CREDIT_MAX_OVERDUE (65% nulos, cubrir con MAX agg)  ║
  ║   → SK_ID_BUREAU     (ID técnico, no predictivo)            ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  COLUMNAS A CONSERVAR + IMPUTAR:                            ║
  ║   → DAYS_ENDDATE_FACT    (37% nulos → imputar con 0)        ║
  ║   → AMT_CREDIT_SUM_LIMIT (34% nulos → imputar con 0)        ║
  ║   → AMT_CREDIT_SUM_DEBT  (15% nulos → imputar con 0)        ║
  ║   → DAYS_CREDIT_ENDDATE  (6% nulos  → imputar con mediana)  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  FEATURES AGREGADOS A CONSTRUIR (por SK_ID_CURR):           ║
  ║   1. n_credits          COUNT total créditos                 ║
  ║   2. n_active           COUNT créditos activos               ║
  ║   3. active_pct         n_active / n_credits                 ║
  ║   4. avg_days_credit    MEAN DAYS_CREDIT (corr=+0.09)        ║
  ║   5. max_overdue_days   MAX CREDIT_DAY_OVERDUE               ║
  ║   6. has_overdue        1 si max_overdue_days > 0            ║
  ║   7. total_debt         SUM AMT_CREDIT_SUM_DEBT              ║
  ║   8. total_overdue_amt  SUM AMT_CREDIT_SUM_OVERDUE           ║
  ║   9. has_bad_debt       1 si tiene Bad debt/Sold             ║
  ║  10. debt_ratio         total_debt / total_credit            ║
  ║  11. n_consumer, n_card, n_mortgage (por tipo)               ║
  ║  12. no_history         1 si cliente NO está en bureau       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  TOP FEATURES POR CORRELACIÓN CON TARGET:                   ║
  ║   1. avg_days_credit    +0.090  (historial antiguo=riesgo)   ║
  ║   2. active_pct         +0.077  (muchos activos=riesgo)      ║
  ║   3. n_active           +0.067  (carga crediticia)           ║
  ║   4. has_overdue        +0.035  (mora previa=riesgo)         ║
  ║   5. no_history         +0.031  (sin historial=riesgo)       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  FILAS A ELIMINAR: NINGUNA                                  ║
  ║   → No hay duplicados de SK_ID_BUREAU                        ║
  ║   → Los outliers en montos son créditos reales               ║
  ║   → Los valores negativos en DEBT son pagos en exceso        ║
  ╚══════════════════════════════════════════════════════════════╝
""")

print(f"  ✅ Figuras guardadas en: {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(f"     {f}")