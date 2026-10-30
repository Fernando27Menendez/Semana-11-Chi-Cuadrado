import numpy as np 
from scipy import stats 

# Paso 1: ingresar la muestra 
tiempos = np.array([8, 9, 10, 10, 11, 12, 12, 13, 14, 15]) 

# Paso 2: declarar parámetros externos de la normal 
media = 11.4 
desv = 2.2 

# Paso 3: ejecutar K-S pasando la función cdf directamente con loc y scale
resultado = stats.kstest(tiempos, stats.norm(loc=media, scale=desv).cdf) 

# Paso 4: mostrar resultados 
print("D:", resultado.statistic) 
print("Valor p:", resultado.pvalue) 

# Paso 5: tomar decisión 
alpha = 0.05 
if resultado.pvalue <= alpha: 
    print("Rechazar H0: evidencia de diferencia") 
else: 
    print("No rechazar H0: evidencia insuficiente para rechazar el ajuste")