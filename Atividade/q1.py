import numpy as np
import os
import pandas as pd

# Só pra achar diretamente o caminho dos CSVs
base_dir = os.path.join(os.path.dirname(__file__), 'CSVs')
iris = pd.read_csv(os.path.join(base_dir, 'iris.csv'))

def process_iris_data(iris, species, pental_lenght):
    # Filtra os dados pela espécie
    iris_species = iris[iris['Species'] == species]
    
    column_lenghts = iris_species[pental_lenght]   # Filtra coluna comprimento da pétala
    array_column_lenght = np.array(column_lenghts) # Transforma a coluna num array
    media_coluna = array_column_lenght.mean()
    print(f"Média de {species}: {media_coluna}")
    return media_coluna

media_setosa = process_iris_data(iris, 'Iris-setosa', 'PetalLengthCm')
media_versicolor = process_iris_data(iris, 'Iris-versicolor', 'PetalLengthCm')
media_virginica = process_iris_data(iris, 'Iris-virginica', 'PetalLengthCm')

medias = {
    'Iris-setosa': media_setosa,
    'Iris-versicolor': media_versicolor,
    'Iris-virginica': media_virginica
}

maior_media_especie = max(medias, key=medias.get)
maior_media_valor = medias[maior_media_especie]

print(f"A espécie com a maior média de comprimento da pétala é {maior_media_especie}"
      f" com uma média de {maior_media_valor:.2f} cm.")
print()
### TRECHOS DE CÓDIGO QUE GERAM ELEMENTOS ADICIONAIS PARA COMPOSIÇÃO DE RELATÓRIO

# Existe uma tendência de aumento no comprimento médio das pétalas das espécies de Iris,
# com a Iris-setosa tendo o menor comprimento médio, seguida pela Iris-versicolor e Iris-virginica.

# 1. Analisar os desvios de cada espécie
# Se o desvio padrão de uma espécie é muito maior que o de outra, 
# pode-se afirmar que há mais variabilidade na espécie com maior desvio padrão.
desvio_setosa = np.std(iris[iris['Species'] == 'Iris-setosa']['PetalLengthCm'])
desvio_versicolor = np.std(iris[iris['Species'] == 'Iris-versicolor']['PetalLengthCm'])
desvio_virginica = np.std(iris[iris['Species'] == 'Iris-virginica']['PetalLengthCm'])

print(f"O desvio padrão do comprimento das pétalas de Iris-setosa é {desvio_setosa:.2f}.")
print(f"O desvio padrão do comprimento das pétalas de Iris-versicolor é {desvio_versicolor:.2f}.")
print(f"O desvio padrão do comprimento das pétalas de Iris-virginica é {desvio_virginica:.2f}.")
print()
# Comparar a média geral (dos comprimentos das pétalas) com cada espécie e observar
# como cada espécie se comporta em relação a média.
def compara_especie(nome_especie, media_especie, media_geral):
    if media_especie > media_geral:
        print(f"A média do comprimento das pétalas da {nome_especie} ({media_especie:.2f}) é superior à média geral ({media_geral:.2f}).")
    else:
        print(f"A média do comprimento das pétalas da {nome_especie} ({media_especie:.2f}) é inferior à média geral ({media_geral:.2f}).")

def compara_media(iris, media_setosa, media_versicolor, media_virginica):
    media_geral = iris['PetalLengthCm'].mean()
    compara_especie("Iris-setosa", media_setosa, media_geral)
    compara_especie("Iris-versicolor", media_versicolor, media_geral)
    compara_especie("Iris-virginica", media_virginica, media_geral)
compara_media(iris, media_setosa, media_versicolor, media_virginica)
print()
# Vale lembrar que no aumento percentual, o cálculo é feito em cima do valor base
# exemplo: média de virginica é 2.8 vezes maior que média setosa
# sendo a média setosa igual a 1.464, média de virginica seria 2,8 * 1.464 + 1.464 (valor base) 
aumento_percentual = ((media_virginica - media_setosa) / media_setosa) * 100
print(f"A maior média do comprimento de pétalas (Iris-virginica) é {aumento_percentual:.2f}% maior que a da menor média (Iris-setosa)")
aumento_percentual = ((media_virginica - media_versicolor) / media_versicolor) * 100
print(f"A maior média do comprimento de pétalas (Iris-virginica) é {aumento_percentual:.2f}% maior que a da segunda maior média (Iris-versicolor)")
