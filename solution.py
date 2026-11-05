import pandas as pd
import numpy as np


def regressao_linear_multipla(arquivo_csv):
  
    df = pd.read_csv(arquivo_csv)
    
    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()
    
    area = np.array(area)
    quartos = np.array(quartos)
    preco = np.array(preco)
    
    n = len(preco)
    

    X = np.column_stack((np.ones(n), area, quartos))
    

    y = preco.reshape(-1, 1)
    
   
    X_transposto = X.T
    beta = np.linalg.inv(X_transposto @ X) @ X_transposto @ y
    
   
    return [float(beta[0, 0]), float(beta[1, 0]), float(beta[2, 0])]


if __name__ == "__main__":
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