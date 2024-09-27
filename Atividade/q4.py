import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Diretório do arquivo CSV
base_dir = os.path.join(os.path.dirname(__file__), 'CSVs')
iris = pd.read_csv(os.path.join(base_dir, 'iris.csv'))

# Definir as colunas de interesse e as espécies
columns_specie = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
species_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Função para calcular a variabilidade (variância) das características
def calcular_variabilidade(iris):
    variancias = {}
    
    # Iterar sobre cada coluna
    for coluna in columns_specie:
        variancias[coluna] = []
        # Iterar sobre cada espécie
        for especie in species_list:
            iris_species = iris[iris['Species'] == especie]
            # Calcular a variância para a coluna específica e espécie
            variancia = np.var(iris_species[coluna], ddof=1)  # ddof=1 para correção da variância amostral
            variancias[coluna].append(variancia)

    return variancias

# Obter as variâncias
variancias = calcular_variabilidade(iris)

# Criar um DataFrame para organizar as variâncias e plotar
variancias_df = pd.DataFrame(variancias, index=species_list).reset_index()
variancias_df = variancias_df.melt(id_vars='index', var_name='Característica', value_name='Variância')
variancias_df.rename(columns={'index': 'Espécie'}, inplace=True)

# Exibir as características com maior variabilidade para cada espécie
for especie in species_list:
    especie_variancias = variancias_df[variancias_df['Espécie'] == especie]
    maior_variabilidade = especie_variancias.loc[especie_variancias['Variância'].idxmax()]
    print(f"{especie}: {maior_variabilidade['Característica']} (Variância: {maior_variabilidade['Variância']:.2f})")

# Plotar o gráfico de barras das variâncias
plt.figure(figsize=(10, 6))
sns.barplot(x='Característica', y='Variância', hue='Espécie', data=variancias_df)
plt.title('Variância das Características das Espécies de Íris')
plt.ylabel('Variância')
plt.xlabel('Características')
plt.xticks(rotation=45)
plt.legend(title='Espécie')
plt.tight_layout()
plt.show()
