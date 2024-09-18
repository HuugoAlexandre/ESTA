import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os # só pra não poluir o path com caminho absoluto
import scipy.stats as scp

sync = np.array([94, 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2, 87.9, 87.9, 93.5, 82.3, 79.3,
78.3, 71.6, 88.6, 74.6, 74.1, 80.6])
asyncr = np.array([77.1, 71.7, 91, 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2])

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
print("ptp sync: ", np.ptp(sync))   # # numpy atualizou, nova sintaxe

print("max assyncr: ", max(asyncr))
print("min assyncr: ", min(asyncr))
print("Max - Min: ", max(asyncr) - min(asyncr)) # igual a asyncr.ptp
print("ptp assyncr: ", np.ptp(asyncr))  # numpy atualizou, nova sintaxe

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

# Coeficiente de variação síncrono e assíncrono
cvs = s1 / mediaSync
print("CVS: ", cvs)
cva = s2 / mediaAsyncr
print("CVA: ", cva)

# plt.hist(dados, beans, range) -> beans tem que ser múltiplo da diferença do range para que o gráfico seja bem plotado 
plt.hist(sync, 5, (65, 95))
plt.show()

plt.hist(asyncr, 5, (65,95)) # 
plt.show()

#boxplot
sns.boxplot(sync)
plt.show()
sns.boxplot(asyncr)
plt.show()

sns.boxplot([sync, asyncr])
plt.xticks([0,1], ['sync', 'asyncr'])
plt.xlabel("Work type")
plt.ylabel("Hours")
plt.show()

base_dir = os.path.join(os.path.dirname(__file__), 'CSVs')

stock = pd.read_csv(os.path.join(base_dir, 'stock_data.csv'))
iris = pd.read_csv(os.path.join(base_dir, 'iris.csv'))

print(iris.head())
sns.boxplot(stock['Open'])
plt.show()

dados = pd.read_csv(os.path.join(base_dir, 'stock_data.csv'))
x = dados['Open'].mean()
ponto_corte = dados['Open'].std(ddof=1) * 3
inf, sup = x - ponto_corte, x + ponto_corte
outliers = dados['Open'][(dados['Open'] < inf) | (dados['Open'] > sup)]
print(outliers)

# Time Series -> Análise da variabilidade de determinado dado em função do tempo
stock['Date'] = pd.to_datetime(stock['Date'])
sns.lineplot(data=stock, x='Date', y='High')
sns.lineplot(data=stock, x='Date', y='Low')
plt.show()
print(stock.head())

x = iris['SepalLengthCm']
y = iris['SepalWidthCm']
# Coeficiente de correlação, saber o quanto uma variável está relacionada uma com a outra
print(scp.spearmanr(x, y)) # assimétrico, não paramétrico
# scp.pearsonr(x,y) -> teste paramétrico (dentro de certo parâmetro), todas as variáveis precisam ter distribuição normal
sns.scatterplot(data=iris, x='SepalLengthCm', y='SepalWidthCm')
plt.show()
sns.boxplot([x,y])
plt.show()

tipo_e_tamanho = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Filtra pelo nome da espécie
iris_setosa = iris[iris['Species'] == 'Iris-setosa']

def process_iris_data(iris, species, x_col, y_col):
    # Filtra os dados pela espécie
    iris_species = iris[iris['Species'] == species]
    
    # Extrai os valores de x e y
    x = iris_species[x_col]
    y = iris_species[y_col]
    
    # Cria o gráfico de dispersão
    sns.scatterplot(data=iris_species, x=x_col, y=y_col)
    
    # Calcula a correlação de Pearson
    correlation, p_value = scp.pearsonr(x, y)
    print(f"Correlação de Pearson entre {x_col} e {y_col} para {species}: {correlation} (p-value: {p_value})")
    
    # Exibe o gráfico
    plt.show()

# Usando a função para vários pares de colunas e espécies
columns_pairs = [
    ('SepalLengthCm', 'SepalWidthCm'),
    ('PetalLengthCm', 'PetalWidthCm'),
    ('SepalLengthCm', 'PetalLengthCm'),
    ('SepalLengthCm', 'PetalWidthCm'),
    ('SepalWidthCm', 'PetalLengthCm'),
    ('SepalWidthCm', 'PetalWidthCm')
]

species_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Loop para processar todas as combinações de espécies e pares de colunas
for species in species_list:
    for x_col, y_col in columns_pairs:
        process_iris_data(iris, species, x_col, y_col)

def heat_map(species):
    name_specie = iris[iris['Species'] == species]
    iris_specie = name_specie.drop(columns=['Id', 'Species'])
    iris_specie.head()
    cormax = iris_specie.corr()
    print(cormax)
    sns.heatmap(cormax, annot=True)
    plt.show()

for specie in species_list:
    heat_map(specie)
