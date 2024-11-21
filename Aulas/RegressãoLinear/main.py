import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import os

diretorio_script = os.path.dirname(os.path.abspath(__file__))

# Concatena o caminho para o arquivo CSV, levando em conta o diretório do script
caminho_csv = os.path.join(diretorio_script, '..', 'CSVs', 'iris.csv')

# Lê o arquivo CSV
df = pd.read_csv(os.path.abspath(caminho_csv))

# Lista das espécies únicas
species_list = df['Species'].unique()

# Limite de R² para considerar a regressão boa, posso mudar se eu quiser
r2_threshold = 0.7

# Loop para fazer a regressão para sépalas e pétalas em cada espécie
for species in species_list:
    # Filtrar os dados para a espécie atual
    subset = df[df['Species'] == species]
   
    # Regressão para sépalas
    X_sepal = subset[['SepalLengthCm']].values  # Variável independente (x)
    y_sepal = subset['SepalWidthCm'].values     # Variável dependente (y)

    model_sepal = LinearRegression()
    model_sepal.fit(X_sepal, y_sepal)               # fit treina o modelo dado o conjunto de dados x e y
    y_sepal_pred = model_sepal.predict(X_sepal)     # predict faz a predição dos dados fornecidos
    r2_sepal = model_sepal.score(X_sepal, y_sepal)  # Calcula o R² para o subset específico
    y_sepal_ssr = np.sum((y_sepal - y_sepal_pred) ** 2)
    # Correlação específica do subset
    corr_sepal = subset[['SepalLengthCm', 'SepalWidthCm']].corr().iloc[0, 1]
   
    print(f"\nEspécie: {species} - Sépala")
    print(f"Correlação de Pearson: {corr_sepal}")
    print(f"Intercepto: {model_sepal.intercept_:.3f}, Coeficiente: {model_sepal.coef_[0]:.3f}")
    print(f"Função de regressão: y = {model_sepal.intercept_:.3f} + {model_sepal.coef_[0]:.3f} * x")
    print(f"Coeficiente de Determinação (R²): {r2_sepal:.2f}")
    print(f"SSR: {y_sepal_ssr}")
   
    if r2_sepal >= r2_threshold:
        print("A regressão para sépalas é suficientemente boa.")
    else:
        print("A regressão para sépalas não é suficientemente boa.")

    # Regressão para pétalas
    X_petal = subset[['PetalLengthCm']].values  # Variável independente (x)
    y_petal = subset['PetalWidthCm'].values     # Variável dependente (y)

    model_petal = LinearRegression()
    model_petal.fit(X_petal, y_petal)
    y_petal_pred = model_petal.predict(X_petal)
    r2_petal = model_petal.score(X_petal, y_petal)  # Calcula o R² para o subset específico
    y_petal_ssr = np.sum((y_petal - y_petal_pred) ** 2)
    # Correlação específica do subset
    corr_petal = subset[['PetalLengthCm', 'PetalWidthCm']].corr().iloc[0, 1]
   
    print(f"\nEspécie: {species} - Pétala")
    print(f"Correlação de Pearson: {corr_petal}")
    print(f"Intercepto: {model_petal.intercept_:.3f}, Coeficiente: {model_petal.coef_[0]:.3f}")
    print(f"Função de regressão: y = {model_petal.intercept_:.3f} + {model_petal.coef_[0]:.3f} * x")
    print(f"Coeficiente de Determinação (R²): {r2_petal:.2f}")
    print(f"SSR: {y_petal_ssr}")

    if r2_petal >= r2_threshold:
        print("A regressão para pétalas é suficientemente boa.")
    else:
        print("A regressão para pétalas não é suficientemente boa.")
