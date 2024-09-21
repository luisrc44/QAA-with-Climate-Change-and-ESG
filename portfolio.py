import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
import pandas as pd

class PortfolioOptimizer:
    def __init__(self, asset_prices, risk_free_rate, average_asset_returns=None):
        """
        Inicializa la clase PortfolioOptimizer con los retornos de los activos y la tasa libre de riesgo.

        :param asset_prices: DataFrame de retornos de los activos (diarios, mensuales, etc.).
        :param risk_free_rate: Tasa libre de riesgo (puede ser diaria o ajustada según la frecuencia de los datos).
        :param average_asset_returns: Retornos promedio de los activos (para cálculos como Sharpe o Sortino).
        """
        self.asset_prices = asset_prices
        self.rf = risk_free_rate
        self.average_asset_returns = average_asset_returns if average_asset_returns is not None else asset_prices.mean()

    def portfolio_return(self, weights):
        """
        Calcula el retorno del portafolio basado en los pesos de los activos.
        
        :param weights: Pesos del portafolio.
        :return: Retorno esperado del portafolio.
        """
        return np.dot(weights, self.average_asset_returns)
    
    def portfolio_variance(self, weights):
        """
        Calcula la varianza del portafolio.
        
        :param weights: Pesos del portafolio.
        :return: Varianza del portafolio.
        """
        return np.dot(weights.T, np.dot(self.asset_prices.cov(), weights))
    
    def negative_sharpe_ratio(self, weights):
        """
        Calcula el Sharpe Ratio negativo para minimizar en la optimización.

        :param weights: Pesos del portafolio.
        :return: Sharpe Ratio negativo.
        """
        portfolio_ret = self.portfolio_return(weights)
        portfolio_vol = np.sqrt(self.portfolio_variance(weights))
        sharpe_ratio = (portfolio_ret - self.rf) / portfolio_vol
        return -sharpe_ratio
    
    def generate_random_portfolios(self, num_portfolios=1000):
        """
        Genera muchos portafolios aleatorios y calcula el Sharpe Ratio para cada uno.
        
        :param num_portfolios: Número de portafolios aleatorios a generar.
        :return: Un DataFrame con los retornos, la volatilidad y el Sharpe Ratio de los portafolios.
        """
        results = np.zeros((3, num_portfolios))
        weights_array = []

        for i in range(num_portfolios):
            # Generar pesos aleatorios y normalizarlos para que sumen 1
            weights = np.random.random(len(self.average_asset_returns))
            weights /= np.sum(weights)
            
            # Calcular retorno y volatilidad
            portfolio_ret = self.portfolio_return(weights)
            portfolio_vol = np.sqrt(self.portfolio_variance(weights))
            sharpe_ratio = (portfolio_ret - self.rf) / portfolio_vol
            
            results[0, i] = portfolio_ret
            results[1, i] = portfolio_vol
            results[2, i] = sharpe_ratio
            weights_array.append(weights)
        
        return pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio']), weights_array

    def plot_random_portfolios(self, num_portfolios=1000):
        """
        Genera portafolios aleatorios y los grafica según su retorno y volatilidad.
        
        :param num_portfolios: Número de portafolios a generar.
        """
        df, weights_array = self.generate_random_portfolios(num_portfolios)

        # Graficar los portafolios generados
        plt.scatter(df['Volatility'], df['Return'], c=df['Sharpe Ratio'], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatilidad')
        plt.ylabel('Retorno Esperado')
        plt.title(f'Portafolios Aleatorios ({num_portfolios})')
        plt.show()
        
        return df, weights_array
