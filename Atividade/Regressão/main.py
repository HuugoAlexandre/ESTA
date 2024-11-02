import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CSVs')
df = pd.read_csv(os.path.join(base_dir, 'Iris.csv'))

species_list = df['Species'].unique()

r2_threshold = 0.7

for species in species_list:
    subset = df[df['Species'] == species]
    
    # Análise para sépalas (Comprimento x Largura)
    X_sepal = subset[['SepalLengthCm']].values  # Variável independente (x) - comprimento da sépala
    y_sepal = subset['SepalWidthCm'].values     # Variável dependente (y) - largura da sépala

    # Calcular a correlação de Pearson para sépalas
    corr_sepal = subset['SepalLengthCm'].corr(subset['SepalWidthCm'])

    # Ajustar o modelo de regressão linear para sépalas
    model_sepal = LinearRegression()
    model_sepal.fit(X_sepal, y_sepal)
    y_sepal_pred = model_sepal.predict(X_sepal)
    r2_sepal = model_sepal.score(X_sepal, y_sepal)

    # Calcular os erros residuais para sépalas
    residuals_sepal = y_sepal - y_sepal_pred
    ssr_sepal = np.sum(residuals_sepal ** 2)  # Soma dos quadrados dos resíduos (SSR)

    # Exibir resultados para a regressão das sépalas
    print(f"\nEspécie: {species} - Sépala")
    print(f"Correlação de Pearson: {corr_sepal:.2f}")
    print(f"Função de Regressão: y = {model_sepal.intercept_:.4f} + {model_sepal.coef_[0]:.4f} * x")
    print(f"Coeficiente de Determinação (R²): {r2_sepal:.2f}")
    print(f"Soma dos Quadrados dos Resíduos (SSR): {ssr_sepal:.4f}")
    
    if r2_sepal >= r2_threshold and abs(corr_sepal) > 0.7:
        print("Conclusão: A regressão para sépalas é suficientemente boa.")
    else:
        print("Conclusão: A regressão para sépalas não é suficientemente boa.")
    
    # Plotar a regressão para sépalas
    plt.figure()
    plt.scatter(X_sepal, y_sepal, color='blue', label='Dados reais')
    plt.plot(X_sepal, y_sepal_pred, color='red', label='Linha de Regressão')
    plt.title(f"Regressão Linear - {species} (Sépala)")
    plt.xlabel("Comprimento da Sépala")
    plt.ylabel("Largura da Sépala")
    plt.legend()
    plt.show()

    # Análise para pétalas (Comprimento x Largura)
    X_petal = subset[['PetalLengthCm']].values  # Variável independente (x) - comprimento da pétala
    y_petal = subset['PetalWidthCm'].values     # Variável dependente (y) - largura da pétala

    # Calcular a correlação de Pearson para pétalas
    corr_petal = subset['PetalLengthCm'].corr(subset['PetalWidthCm'])

    # Ajustar o modelo de regressão linear para pétalas
    model_petal = LinearRegression()
    model_petal.fit(X_petal, y_petal)
    y_petal_pred = model_petal.predict(X_petal)
    r2_petal = model_petal.score(X_petal, y_petal)

    # Calcular os erros residuais para pétalas
    residuals_petal = y_petal - y_petal_pred
    ssr_petal = np.sum(residuals_petal ** 2)  # Soma dos quadrados dos resíduos (SSR)

    # Exibir resultados para a regressão das pétalas
    print(f"\nEspécie: {species} - Pétala")
    print(f"Correlação de Pearson: {corr_petal:.2f}")
    print(f"Função de Regressão: y = {model_petal.intercept_:.4f} + {model_petal.coef_[0]:.4f} * x")
    print(f"Coeficiente de Determinação (R²): {r2_petal:.2f}")
    print(f"Soma dos Quadrados dos Resíduos (SSR): {ssr_petal:.4f}")
    
    # Avaliar se a regressão é suficientemente boa
    if r2_petal >= r2_threshold and abs(corr_petal) > 0.7:
        print("Conclusão: A regressão para pétalas é suficientemente boa.")
    else:
        print("Conclusão: A regressão para pétalas não é suficientemente boa.")
    
    # Plotar a regressão para pétalas
    plt.figure()
    plt.scatter(X_petal, y_petal, color='blue', label='Dados reais')
    plt.plot(X_petal, y_petal_pred, color='red', label='Linha de Regressão')
    plt.title(f"Regressão Linear - {species} (Pétala)")
    plt.xlabel("Comprimento da Pétala")
    plt.ylabel("Largura da Pétala")
    plt.legend()
    plt.show()
