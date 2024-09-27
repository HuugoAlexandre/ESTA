import os
import pandas as pd
import numpy as np
import statistics as sts
import seaborn as sns
import matplotlib.pyplot as plt

# Só pra achar diretamente o caminho dos CSVs
base_dir = os.path.join(os.path.dirname(__file__), 'CSVs')
iris = pd.read_csv(os.path.join(base_dir, 'iris.csv'))

columns_specie = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
species_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def calcula_distribuicao(iris, especie, coluna):
    # Filtra os dados pela espécie
    iris_species = iris[iris['Species'] == especie]
    coluna_especie = iris_species[coluna]

    coluna_array = np.array(coluna_especie)
    
    # Cálculo das estatísticas
    media = coluna_array.mean()
    moda = sts.multimode(coluna_especie)
    mediana = np.median(coluna_array)
    desvio = np.std(coluna_especie, ddof=1)
    amplitude = max(coluna_array) - min(coluna_array)
    
    inf, sup = media - desvio * 3, media + desvio * 3
    outliers = list(coluna_especie[(coluna_array < inf) | (coluna_array > sup)])
    
    print(f"Média da coluna {coluna} da espécie {especie}: {media}")
    print(f"Moda da coluna {coluna} da espécie {especie}: {moda}")
    print(f"Mediana da coluna {coluna} da espécie {especie}: {mediana}")
    print(f"Desvio padrão da coluna {coluna} da espécie {especie}: {desvio}")
    print(f"Amplitude da coluna {coluna} da espécie {especie}: {amplitude}")
    print(f"Outliers da coluna {coluna} da espécie {especie}: {len(outliers)} encontrados ->  {outliers}")
    print()

    # Plotando a distribuição
    plt.figure(figsize=(8, 4))
    sns.histplot(coluna_especie, bins=10, kde=True)
    plt.title(f'Distribuição da {coluna} - {especie}')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')
    plt.axvline(media, color='r', linestyle='dashed', linewidth=1, label='Média')
    plt.axvline(mediana, color='g', linestyle='dashed', linewidth=1, label='Mediana')
    if moda:  # Verifica se há modas
        plt.axvline(moda[0], color='b', linestyle='dashed', linewidth=1, label='Moda')
    plt.legend()
    plt.grid(True)
    plt.show()

# Executar a função para todas as espécies e colunas
for especie in species_list:
    for coluna in columns_specie:
        calcula_distribuicao(iris, especie, coluna)
