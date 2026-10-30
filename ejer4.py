import numpy as np 
from scipy.stats import chi2 
observadas = np.array([8, 18, 20, 10, 4], dtype=float) 
esperadas = np.array([6, 12, 18, 15, 9], dtype=float) 
# Paso 1: contribución de cada categoría 
contribuciones = (observadas - esperadas)**2 / esperadas 
# Paso 2: sumar contribuciones 
estadistico = contribuciones.sum() 
# Paso 3: grados de libertad (sin parámetros estimados) 
gl = len(observadas) - 1 
# Paso 4: valor p de cola derecha 
valor_p = chi2.sf(estadistico, gl) 
print("Contribuciones:", contribuciones) 
print("Chi-cuadrado:", estadistico) 
print("Grados de libertad:", gl) 
print("Valor p:", valor_p)
if valor_p <= 0.05: 
    print("Se rechaza H0") 
else: 
    print("No se rechaza H0")