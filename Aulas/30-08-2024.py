import numpy as np
import statistics as st

sync = np.array([94, 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2, 87.9, 87.9, 93.5, 82.3, 79.3,
78.3, 71.6, 88.6, 74.6, 74.1, 80.6])
asyncr = np.array([77.1, 71.7, 91, 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2, 78.2])

# Média
mediaSync = sync.mean() 
print("Média Sync: ", mediaSync)
mediaAsyncr = asyncr.mean()
print("Média Async: ", mediaAsyncr)

# Mediana
medianaSync = np.median(sync)
print("Mediana Sync: ", medianaSync)
medianaAsyncr = np.median(asyncr)
print("Mediana Async: ", medianaAsyncr)

# q para quartil, p para percentil (percentil por padrão é de 10 em 10 porcento, quartil de 25 em 25)
q1s = np.percentile(sync, 25)
q1a = np.percentile(asyncr, 25)
p3s = np.percentile(sync, 30)
q3a = np.percentile(asyncr, 75)
q3s = np.percentile(sync, 75)
print("Primeiro quartil Sync: ", q1s)
print("Terceiro quartil Sync: ", q3s)
print("Primeiro quartil asyncr: ", q1a)
print("Terceiro quartil asyncr: ", q3a)

# Moda
modaSync = st.multimode(sync)
print("Moda Sync: ", modaSync)
modaAsyncr = st.multimode(asyncr)
print("Moda Async: ", modaAsyncr)

# Amplitude
print("max sync: ", max(sync))
print("min sync: ", min(sync))
print("Max - Min: ", max(sync) - min(sync)) # igual a sync.ptp
print("ptp sync: ", sync.ptp())

print("max assyncr: ", max(asyncr))
print("min assyncr: ", min(asyncr))
print("Max - Min: ", max(asyncr) - min(asyncr)) # igual a asyncr.ptp
print("ptp assyncr: ", asyncr.ptp())

q1 = np.percentile(sync, 25)
q3 = np.percentile(sync, 75)
iqr = q3 - q1
print("IQR do sync: ", iqr)

q1 = np.percentile(asyncr, 25)
q3 = np.percentile(asyncr, 75)
iqr = q3 - q1
print("IQR do asyncr: ", iqr)

# Variância
var1 = sync.var(ddof=1) # ddof = 1 simboliza a amostra
print("Variância Sync: ", var1)
var2 = asyncr.var(ddof=1)
print("Variância Assyncr: ", var2)

# Desvio Padrão
s1 = sync.std(ddof=1)
print("Desvio padrão do sync: ", s1)
s2 = asyncr.std(ddof=1)
print("Desvio padrão do asyncr: ", s2)