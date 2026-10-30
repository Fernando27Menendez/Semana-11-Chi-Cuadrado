"""
Ejercicio 6 (Semana 11): decidir qué prueba corresponde.
- Tiempos entre llegadas (cuantitativa continua) -> Kolmogórov–Smirnov
- Tipo de trámite (categórica)                   -> Chi-cuadrado
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
rng = np.random.default_rng(123)

# ------------------------------------------------------------------
# 1. Cargar datos (120 tiempos y 120 tipos de trámite)
#    Si no existe el CSV, se simulan datos ILUSTRATIVOS: reemplazar por los reales.
# ------------------------------------------------------------------
ARCHIVO = "datos_centro_atencion.csv"   # columnas: tiempo_entre_llegadas, tipo_tramite
if os.path.exists(ARCHIVO):
    df = pd.read_csv(ARCHIVO)
else:
    df = pd.DataFrame({
        "tiempo_entre_llegadas": np.round(rng.exponential(5.0, 120), 2),
        "tipo_tramite": rng.choice(["Consulta", "Pago", "Reclamo", "Otros"],
                                   size=120, p=[0.30, 0.35, 0.20, 0.15]),
    })

# ------------------------------------------------------------------
# 2. Paso 1: separar variables y decidir la prueba según el tipo
# ------------------------------------------------------------------
def seleccionar_prueba(serie: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(serie):
        return "K-S (u otra prueba de ajuste)"
    return "Chi-cuadrado"

for col in df.columns:
    print(f"{col:25s} -> {seleccionar_prueba(df[col])}")

# ------------------------------------------------------------------
# 3. Tiempos entre llegadas: descriptivos + K-S con exponencial
# ------------------------------------------------------------------
x = df["tiempo_entre_llegadas"].dropna().to_numpy()
n = len(x)
print("\n--- Tiempos entre llegadas ---")
print(pd.Series(x).describe())

# Paso 3: λ se ESTIMA con la misma muestra -> el p clásico no es válido
lam = 1 / x.mean()
ks = stats.kstest(x, "expon", args=(0, 1 / lam))
D_obs = ks.statistic
print(f"λ estimado = {lam:.4f} | D = {D_obs:.4f} | p clásico (solo referencial) = {ks.pvalue:.4f}")

# Bootstrap paramétrico: simular Exp(λ̂), re-estimar λ en cada réplica y calcular D*
def estadistico_D(muestras):
    """D de K-S contra Exp(media muestral) para cada fila de 'muestras'."""
    s = np.sort(muestras, axis=1)
    m = s.shape[1]
    F0 = 1 - np.exp(-s / s.mean(axis=1, keepdims=True))
    i = np.arange(1, m + 1)
    d_mas = (i / m - F0).max(axis=1)
    d_menos = (F0 - (i - 1) / m).max(axis=1)
    return np.maximum(d_mas, d_menos)

B = 5000
D_boot = estadistico_D(rng.exponential(1 / lam, size=(B, n)))
p_boot = (np.sum(D_boot >= D_obs) + 1) / (B + 1)
dec_ks = "Rechazar H0" if p_boot <= ALPHA else "No rechazar H0"
print(f"p bootstrap = {p_boot:.4f} -> {dec_ks}")

# ------------------------------------------------------------------
# 4. Tipo de trámite: tabla de frecuencias + Chi-cuadrado
# ------------------------------------------------------------------
print("\n--- Tipo de trámite ---")
# Proporciones históricas JUSTIFICADAS (definirlas por nombre evita errores de orden)
prop_hist = pd.Series({"Consulta": 0.30, "Pago": 0.35, "Reclamo": 0.20, "Otros": 0.15})

obs = df["tipo_tramite"].value_counts().reindex(prop_hist.index, fill_value=0)
esp = prop_hist * obs.sum()                     # E = N × proporción

assert np.isclose(prop_hist.sum(), 1.0), "Las proporciones deben sumar 1"
if (esp < 5).any():
    print("Aviso: hay esperadas < 5; reagrupar categorías con criterio.")

tabla = pd.DataFrame({"Observada": obs, "Esperada": esp,
                      "Contribución": (obs - esp) ** 2 / esp})
print(tabla.round(4))

m = 0                                           # parámetros estimados con la muestra
gl = len(obs) - 1 - m
chi = stats.chisquare(f_obs=obs.to_numpy(), f_exp=esp.to_numpy(), ddof=m)
dec_chi = "Rechazar H0" if chi.pvalue <= ALPHA else "No rechazar H0"
print(f"χ² = {chi.statistic:.4f} | gl = {gl} | p = {chi.pvalue:.4f} -> {dec_chi}")

# ------------------------------------------------------------------
# 5. Tabla resumen
# ------------------------------------------------------------------
resumen = pd.DataFrame({
    "variable": ["Tiempo entre llegadas", "Tipo de trámite"],
    "distribucion": ["Exponencial (λ estimado)", "Proporciones históricas"],
    "prueba": ["K-S + bootstrap paramétrico", "Chi-cuadrado"],
    "estadistico": [D_obs, chi.statistic],
    "gl": ["No aplica", gl],
    "valor_p": [p_boot, chi.pvalue],
    "decision": [dec_ks, dec_chi],
})
print("\n", resumen.to_string(index=False))
resumen.to_csv("resumen_ejercicio6.csv", index=False)