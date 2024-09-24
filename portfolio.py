import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, asset_prices, risk_free_rate, benchmark_prices=None, economic_factors=None, climate_factors=None):
        """
        Inicializa la clase PortfolioOptimizer con los retornos de los activos, la tasa libre de riesgo,
        y los factores económicos/climáticos.

        :param asset_prices: DataFrame de retornos de los activos.
        :param risk_free_rate: Tasa libre de riesgo.
        :param benchmark_prices: Retornos del benchmark (si se utiliza para Omega Ratio, por ejemplo).
        :param economic_factors: Factores económicos para ajustar retornos.
        :param climate_factors: Factores climáticos para ajustar retornos.
        """
        self.asset_prices = asset_prices.select_dtypes(include=[np.number])
        self.rf = risk_free_rate
        self.benchmark_prices = benchmark_prices
        self.economic_factors = economic_factors
        self.climate_factors = climate_factors
        self.average_asset_prices = self.asset_prices.mean()

    def calculate_beta(self, portfolio_returns):
        benchmark_returns = self.benchmark_prices['^GSPC'].values  # Usar el S&P 500 como benchmark
        portfolio_returns = portfolio_returns.reshape(-1, 1)
        benchmark_returns = benchmark_returns.reshape(-1, 1)
        reg = LinearRegression().fit(benchmark_returns, portfolio_returns)
        beta = reg.coef_[0][0]
        return beta

    def calculate_jensen_alpha(self, weights):
        portfolio_returns = np.dot(self.asset_prices, weights)
        portfolio_avg_return = np.mean(portfolio_returns)
        benchmark_avg_return = np.mean(self.benchmark_prices['^GSPC'])
        beta = self.calculate_beta(portfolio_returns)
        jensen_alpha = (portfolio_avg_return - self.rf) - beta * (benchmark_avg_return - self.rf)
        return jensen_alpha

    def calculate_sharpe_ratio(self, weights):
        portfolio_returns = np.dot(weights, self.asset_prices.mean())
        portfolio_variance = np.dot(weights.T, np.dot(self.asset_prices.cov(), weights))
        sharpe_ratio = (portfolio_returns - self.rf) / np.sqrt(portfolio_variance)
        return sharpe_ratio

    def calculate_adjusted_return(self, weights):
        """
        Calcula el retorno ajustado del portafolio usando factores económicos y climáticos,
        considerando los retornos de cada activo en cada periodo de tiempo.
        """
        portfolio_returns = np.dot(self.asset_prices, weights)
        if self.economic_factors is not None and self.climate_factors is not None:
            economic_adjustments = np.dot(self.economic_factors, weights)
            climate_adjustments = np.dot(self.climate_factors, weights)
            adjusted_returns = portfolio_returns + economic_adjustments + climate_adjustments
        else:
            adjusted_returns = portfolio_returns
        return adjusted_returns

    def objective_function(self, weights):
        """
        Función objetivo para la optimización usando mínimos cuadrados.
        """
        adjusted_returns = self.calculate_adjusted_return(weights)
        return np.sum((adjusted_returns - np.mean(adjusted_returns))**2)  # Error cuadrático

    def optimize_with_least_squares(self):
        """
        Optimiza los pesos del portafolio minimizando el error cuadrático con los factores climáticos y económicos.
        """
        num_assets = len(self.asset_prices.columns)
        initial_weights = np.ones(num_assets) / num_assets  # Inicialización con pesos iguales
        bounds = [(0, 1) for _ in range(num_assets)]  # Pesos entre 0 y 1
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # La suma de los pesos debe ser 1

        result = minimize(self.objective_function, initial_weights, bounds=bounds, constraints=constraints)

        return result.x  # Pesos optimizados

    def optimize_with_multiple_strategies(self, num_portfolios=1000, strategies=['sharpe', 'omega', 'sortino']):
        """
        Optimiza los portafolios para múltiples estrategias y devuelve los resultados.

        :param num_portfolios: Número de portafolios a generar.
        :param strategies: Lista de estrategias a utilizar ('sharpe', 'omega', 'sortino').
        :return: Diccionario de portafolios óptimos para cada estrategia.
        """
        optimal_portfolios = {}
        all_portfolios = []  # Para almacenar todos los portafolios y luego aplicar el ranking

        for strategy in strategies:
            portfolio_stats = []
            
            for _ in range(num_portfolios):
                # Optimización por mínimos cuadrados
                weights = self.optimize_with_least_squares()

                # Calcular la métrica basada en la estrategia elegida
                if strategy == 'sharpe':
                    score = self.calculate_sharpe_ratio(weights)
                elif strategy == 'omega':
                    score = self.neg_omega_ratio(weights)
                elif strategy == 'sortino':
                    score = self.neg_sortino_ratio(weights)
                
                # Calcular el retorno ajustado usando factores climáticos/económicos
                adjusted_return = self.calculate_adjusted_return(weights)

                portfolio_stats.append((weights, score, adjusted_return))
                all_portfolios.append((strategy, weights, score, adjusted_return))
            
            ranked_portfolios = sorted(portfolio_stats, key=lambda x: x[1], reverse=True)
            optimal_portfolios[strategy] = ranked_portfolios

        # Crear un DataFrame con todos los portafolios para el ranking
        portfolios_df = pd.DataFrame(all_portfolios, columns=['strategy', 'weights', 'score', 'adjusted_return'])
        
        # Ordenar todos los portafolios por el score en orden descendente (mejor primero)
        portfolios_df['rank'] = portfolios_df['score'].rank(ascending=False)
        portfolios_df = portfolios_df.sort_values(by='score', ascending=False)

        return optimal_portfolios, portfolios_df

