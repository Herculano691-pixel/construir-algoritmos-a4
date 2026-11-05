import pandas as pd
import numpy as np


def regressao_linear_multipla(arquivo_csv):
    """
    Realiza regressão linear múltipla para estimar preço de imóveis.
    
    Modelo: Preco = β0 + β1*Area + β2*Quartos
    
    Args:
        arquivo_csv: Caminho para o arquivo CSV com os dados
        
    Returns:
        Lista com 3 valores: [β0 (intercepto), β1 (coef. area), β2 (coef. quartos)]
    """
    # Ler o arquivo CSV
    df = pd.read_csv(arquivo_csv)
    
    # Extrair as colunas necessárias
    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()
    
    # Converter para arrays numpy
    area = np.array(area)
    quartos = np.array(quartos)
    preco = np.array(preco)
    
    # Número de observações
    n = len(preco)
    
    # Construir a matriz de design X (com coluna de 1s para o intercepto)
    # X = [1, area, quartos]
    X = np.column_stack((np.ones(n), area, quartos))
    
    # Vetor alvo y
    y = preco.reshape(-1, 1)
    
    # Calcular os coeficientes usando a fórmula dos mínimos quadrados:
    # β = (X^T X)^(-1) X^T y
    X_transposto = X.T
    beta = np.linalg.inv(X_transposto @ X) @ X_transposto @ y
    
    # Retornar como lista: [intercepto, coef_area, coef_quartos]
    return [float(beta[0, 0]), float(beta[1, 0]), float(beta[2, 0])]


if __name__ == "__main__":
    # Executa a regressão com o arquivo do projeto
    arquivo = "imoveis_lar_perfeito.csv"
    resultado = regressao_linear_multipla(arquivo)
    
    print("=" * 60)
    print("MODELO DE REGRESSAO LINEAR MULTIPLA - LAR PERFEITO")
    print("=" * 60)
    print(f"Intercepto (beta0):     {resultado[0]:>15.2f}")
    print(f"Coef. Area (beta1):     {resultado[1]:>15.2f}")
    print(f"Coef. Quartos (beta2):  {resultado[2]:>15.2f}")
    print("-" * 60)
    print(f"\nEQUACAO FINAL:")
    print(f"Preco = {resultado[0]:.2f} + {resultado[1]:.2f}*Area + {resultado[2]:.2f}*Quartos")
    print("=" * 60)