import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
        # Eliminar cualquier columna no numérica (por ejemplo, Date) para evitar errores en los cálculos
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

    def neg_omega_ratio(self, weights):
        portfolio_returns = np.dot(self.asset_prices, weights)
        benchmark_returns = self.benchmark_prices['^GSPC'].values  
        excess_returns = portfolio_returns - benchmark_returns
        positive_excess = excess_returns[excess_returns > 0].sum()
        negative_excess = -excess_returns[excess_returns < 0].sum()
        if negative_excess == 0:
            return np.inf
        omega_ratio = positive_excess / negative_excess
        return -omega_ratio

    def neg_sortino_ratio(self, weights):
        portfolio_return = np.dot(weights, self.average_asset_prices)
        excess_returns = self.asset_prices - self.rf
        negative_excess_returns = excess_returns[excess_returns < 0]
        weighted_negative_excess_returns = negative_excess_returns.multiply(weights, axis=1)
        semivariance = np.mean(np.square(weighted_negative_excess_returns.sum(axis=1)))
        if semivariance == 0:
            return np.inf
        sortino_ratio = (portfolio_return - self.rf) / np.sqrt(semivariance)
        return -sortino_ratio

    def calculate_adjusted_return(self, weights):
        """
        Calcula el retorno ajustado del portafolio usando factores económicos y climáticos,
        considerando los retornos de cada activo en cada periodo de tiempo.

        :param weights: Pesos de los activos en el portafolio.
        :return: Retorno ajustado por periodo de tiempo.
        """
        # Calcular los retornos del portafolio dinámicamente para cada periodo de tiempo
        portfolio_returns = np.dot(self.asset_prices, weights)

        # Verificar si hay factores económicos y climáticos
        if self.economic_factors is not None and self.climate_factors is not None:
            # Ajustar los retornos para cada periodo usando los factores
            economic_adjustments = np.dot(self.economic_factors, weights)
            climate_adjustments = np.dot(self.climate_factors, weights)
            
            # Ajustar los retornos del portafolio sumando el impacto de los factores
            adjusted_returns = portfolio_returns + economic_adjustments + climate_adjustments
        else:
            # Si no hay factores climáticos/económicos, mantener el retorno sin ajuste
            adjusted_returns = portfolio_returns

        # Retornar el retorno ajustado en cada periodo
        return adjusted_returns


    def optimize_with_ranking(self, num_portfolios=1000, strategy='sharpe'):
        """
        Optimiza los portafolios y asigna un ranking basado en la estrategia especificada.

        :param num_portfolios: Número de portafolios a generar.
        :param strategy: Estrategia para la optimización ('sharpe', 'omega', 'sortino').
        :return: Lista de portafolios ordenados y sus ponderaciones inversas.
        """
        portfolio_stats = []
        
        for _ in range(num_portfolios):
            weights = np.random.random(len(self.asset_prices.columns))
            weights /= np.sum(weights)
            
            # Calcular la métrica basada en la estrategia elegida
            if strategy == 'sharpe':
                score = self.calculate_sharpe_ratio(weights)
            elif strategy == 'omega':
                score = self.neg_omega_ratio(weights)
            elif strategy == 'sortino':
                score = self.neg_sortino_ratio(weights)
            
            portfolio_stats.append((weights, score))
        
        ranked_portfolios = sorted(portfolio_stats, key=lambda x: x[1], reverse=True)
        total_portfolios = len(ranked_portfolios)
        inverse_weights = [(i+1)/total_portfolios for i in range(total_portfolios)]
        
        return ranked_portfolios, inverse_weights

    def optimize_with_multiple_strategies(self, num_portfolios=1000, strategies=['sharpe', 'omega', 'sortino']):
        """
        Optimiza los portafolios para múltiples estrategias y devuelve los resultados.

        :param num_portfolios: Número de portafolios a generar.
        :param strategies: Lista de estrategias a utilizar ('sharpe', 'omega', 'sortino').
        :return: Diccionario de portafolios óptimos para cada estrategia.
        """
        optimal_portfolios = {}

        for strategy in strategies:
            portfolio_stats = []
            
            for _ in range(num_portfolios):
                weights = np.random.random(len(self.asset_prices.columns))
                weights /= np.sum(weights)
                
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
            
            ranked_portfolios = sorted(portfolio_stats, key=lambda x: x[1], reverse=True)
            optimal_portfolios[strategy] = ranked_portfolios

        return optimal_portfolios

