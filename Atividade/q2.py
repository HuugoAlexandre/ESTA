import os
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.join(os.path.dirname(__file__), 'CSVs')
iris = pd.read_csv(os.path.join(base_dir, 'iris.csv'))

def verificar_normalidade(iris_specie, nome_especie):
    # Extrai os dados da sépala e da pétala
    dados_sep = iris_specie['SepalLengthCm']
    dados_pet = iris_specie['PetalLengthCm']
    
    # Realiza o teste de Shapiro-Wilk
    estatistica_sep, p_valor_sep = stats.shapiro(dados_sep)
    estatistica_pet, p_valor_pet = stats.shapiro(dados_pet)

    print(f"Resultados para {nome_especie}:")
    print(f"--- Sépala ---")
    print(f"Estatística do teste: {estatistica_sep}, p-valor: {p_valor_sep}")
    if p_valor_sep > 0.05:
        print("Distribuição não rejeitada como normal para o comprimento da sépala.")
    else:
        print("Distribuição rejeitada como normal para o comprimento da sépala.")

    print(f"--- Pétala ---")
    print(f"Estatística do teste: {estatistica_pet}, p-valor: {p_valor_pet}")
    if p_valor_pet > 0.05:
        print("Distribuição não rejeitada como normal para o comprimento da pétala.")
    else:
        print("Distribuição rejeitada como normal para o comprimento da pétala.")

    # Estatísticas descritivas
    print("\nEstatísticas descritivas:")
    print(f"Comprimento da Sépala - Média: {np.mean(dados_sep):.2f}, Mediana: {np.median(dados_sep):.2f}, Assimetria: {stats.skew(dados_sep):.2f}, Curtose: {stats.kurtosis(dados_sep):.2f}")
    print(f"Comprimento da Pétala - Média: {np.mean(dados_pet):.2f}, Mediana: {np.median(dados_pet):.2f}, Assimetria: {stats.skew(dados_pet):.2f}, Curtose: {stats.kurtosis(dados_pet):.2f}")

    # Cálculo da correlação
    correlation = iris_specie[['SepalLengthCm', 'PetalLengthCm']].corr().iloc[0, 1]
    print(f"Coeficiente de correlação entre Comprimento da Sépala e Comprimento da Pétala: {correlation:.2f}")

    # 3. Gráfico de Dispersão
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=iris_specie, x='SepalLengthCm', y='PetalLengthCm', hue='Species')
    plt.title('Gráfico de Dispersão: Comprimento da Sépala vs. Comprimento da Pétala')
    plt.show()

    # 5. Heatmap de Correlação
    plt.figure(figsize=(8, 6))
    correlation_matrix = iris_specie[['SepalLengthCm', 'PetalLengthCm']].corr()  # Seleciona apenas colunas numéricas
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap de Correlação')
    plt.show()

def filtra_especie(especie):    
    iris_specie = iris[iris['Species'] == especie]
    verificar_normalidade(iris_specie, especie)

especies = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
for especie in especies:
    filtra_especie(especie)
