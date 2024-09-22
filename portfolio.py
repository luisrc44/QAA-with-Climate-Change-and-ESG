import numpy as np
import scipy.optimize as sco
import pandas as pd


class PortfolioOptimizer:
    def __init__(self, asset_returns, risk_free_rate, average_asset_returns=None):
        """
        Inicializa la clase PortfolioOptimizer con los retornos de los activos y la tasa libre de riesgo.

        :param asset_returns: DataFrame de retornos de los activos.
        :param risk_free_rate: Tasa libre de riesgo.
        :param average_asset_returns: Retornos promedio de los activos (para cálculos como Sharpe o Sortino).
        """
        self.asset_returns = asset_returns
        self.rf = risk_free_rate
        self.average_asset_returns = average_asset_returns if average_asset_returns is not None else self.asset_returns.mean()

    def neg_omega_ratio(self, weights, benchmark_returns, Smart=False):
        """
        Calcula el Omega Ratio negativo para un conjunto de pesos de activos, comparando los retornos diarios con un benchmark diario.
        
        :param weights: Pesos de los activos en el portafolio.
        :param benchmark_returns: Serie con los retornos del benchmark.
        :param Smart: Bandera opcional para aplicar una penalización por autocorrelación.
        :return: Omega Ratio negativo.
        """
        portfolio_returns = np.dot(self.asset_returns, weights)
        excess_returns = portfolio_returns - benchmark_returns

        positive_excess = excess_returns[excess_returns > 0].sum()
        negative_excess = -excess_returns[excess_returns < 0].sum()

        if negative_excess == 0:
            return np.inf

        omega_ratio = positive_excess / negative_excess
        return -omega_ratio

    def neg_sortino_ratio(self, weights):
        """  
        Calcula el Sortino Ratio negativo del portafolio, enfocándose en el riesgo a la baja respecto a la tasa libre de riesgo.
        
        :param weights: Pesos de los activos en el portafolio.
        :return: Sortino Ratio negativo para minimizar en la optimización.
        """
        portfolio_return = np.dot(weights, self.average_asset_returns)
        excess_returns = self.asset_returns - self.rf
        negative_excess_returns = excess_returns[excess_returns < 0]
        weighted_negative_excess_returns = negative_excess_returns.multiply(weights, axis=1)
        semivariance = np.mean(np.square(weighted_negative_excess_returns.sum(axis=1)))

        if semivariance == 0:
            return np.inf

        sortino_ratio = (portfolio_return - self.rf) / np.sqrt(semivariance)
        return -sortino_ratio

