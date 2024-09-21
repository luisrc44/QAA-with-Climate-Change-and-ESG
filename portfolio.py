import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
import pandas as pd

class PortfolioOptimizer:
    def __init__(self, asset_returns, benchmark_returns, risk_free_rate, average_asset_returns=None):
        """
        Inicializa la clase PortfolioOptimizer con los retornos de los activos y el benchmark.

        :param asset_returns: DataFrame de retornos de los activos (diarios, mensuales, etc.).
        :param benchmark_returns: Retornos del benchmark (p.ej., un índice de referencia o tasa libre de riesgo).
        :param risk_free_rate: Tasa libre de riesgo (puede ser diaria o ajustada según la frecuencia de los datos).
        :param average_asset_returns: Retornos promedio de los activos (para cálculos como Sharpe o Sortino).
        """
        self.asset_returns = asset_returns
        self.benchmark_returns = benchmark_returns
        self.rf = risk_free_rate
        self.average_asset_returns = average_asset_returns if average_asset_returns is not None else asset_returns.mean()

    def portfolio_return(self, weights):
        """
        Calcula el retorno del portafolio basado en los pesos de los activos.
        
        :param weights: Pesos del portafolio.
        :return: Retorno esperado del portafolio.
        """
        return np.dot(weights, self.average_asset_returns)
    
    def neg_sortino_ratio(self, weights):
        """
        Calcula el Sortino Ratio negativo, enfocándose en la volatilidad negativa (semivarianza).
        
        :param weights: Pesos del portafolio.
        :return: Sortino Ratio negativo para minimizar en la optimización.
        """
        portfolio_return = np.dot(weights, self.average_asset_returns)
        excess_returns = self.asset_returns - self.rf  # Retornos en exceso sobre la tasa libre de riesgo
        negative_excess_returns = excess_returns[excess_returns < 0]  # Solo los retornos negativos
        weighted_negative_excess_returns = negative_excess_returns.multiply(weights, axis=1)  # Aplicar pesos
        semivariance = np.mean(np.square(weighted_negative_excess_returns.sum(axis=1)))  # Calcular semivarianza

        # Calcular el Sortino Ratio: (Retorno Portafolio - Tasa Libre de Riesgo) / Raíz de la Semivarianza
        sortino_ratio = (portfolio_return - self.rf) / np.sqrt(semivariance)
        
        return -sortino_ratio
    
    def neg_omega_ratio(self, weights, Smart=False):
        """
        Calcula el Omega Ratio negativo comparando retornos diarios del portafolio con un benchmark.

        :param weights: Pesos del portafolio.
        :param Smart: Bandera opcional para aplicar una penalización de autocorrelación (no esencial para este ejercicio).
        :return: Omega Ratio negativo para la optimización.
        """
        # Retornos ponderados del portafolio
        portfolio_returns = pd.DataFrame(self.asset_returns.dot(weights))
        
        # Excesos de retorno sobre el benchmark
        excess_returns = portfolio_returns[0] - self.benchmark_returns[self.benchmark_returns.columns[0]] 
        
        # Ganancias y pérdidas (exceso de retornos positivos y negativos)
        positive_excess = excess_returns[excess_returns > 0].sum()
        negative_excess = -excess_returns[excess_returns < 0].sum()
        
        omega_ratio = positive_excess / negative_excess  # Calcular el Omega Ratio
        
        # Penalización opcional por autocorrelación
        if Smart:
            autocorr_penalty = self.portfolio_autocorr_penalty(weights)
            omega_ratio /= (1 + ((autocorr_penalty - 1) * 2))
        
        return -omega_ratio

    def optimize_portfolio(self, strategy="sharpe"):
        """
        Optimiza el portafolio utilizando la estrategia indicada (Sharpe, Sortino, Omega).
        
        :param strategy: Estrategia de optimización ("sharpe", "sortino", "omega").
        :return: Pesos óptimos del portafolio.
        """
        # Restricciones: la suma de los pesos debe ser 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Fronteras: pesos entre 0 y 1 (no vendemos activos a corto)
        bounds = tuple((0, 1) for _ in range(len(self.average_asset_returns)))
        
        # Peso inicial para la optimización (pesos igual distribuidos)
        initial_weights = len(self.average_asset_returns) * [1. / len(self.average_asset_returns)]
        
        # Selección de la estrategia
        if strategy == "sharpe":
            objective_function = self.negative_sharpe_ratio
        elif strategy == "sortino":
            objective_function = self.neg_sortino_ratio
        elif strategy == "omega":
            objective_function = self.neg_omega_ratio
        else:
            raise ValueError("Estrategia no reconocida. Usa 'sharpe', 'sortino' o 'omega'.")
        
        # Optimización
        opt_results = sco.minimize(objective_function, initial_weights, 
                                   method='SLSQP', bounds=bounds, constraints=constraints)
        
        return opt_results.x

