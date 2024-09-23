import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class PortfolioOptimizer:
    def __init__(self, asset_prices, risk_free_rate, benchmark_prices=None):
        """
        Inicializa la clase PortfolioOptimizer con los retornos de los activos y la tasa libre de riesgo.

        :param asset_prices: DataFrame de retornos de los activos.
        :param risk_free_rate: Tasa libre de riesgo.
        :param benchmark_prices: Retornos del benchmark (si se utiliza para Omega Ratio, por ejemplo).
        """
        # Eliminar cualquier columna no numérica (por ejemplo, Date) para evitar errores en los cálculos
        self.asset_prices = asset_prices.select_dtypes(include=[np.number])
        self.rf = risk_free_rate
        self.benchmark_prices = benchmark_prices
        self.average_asset_prices = self.asset_prices.mean()

    def calculate_beta(self, portfolio_returns):
        """
        Calcula el beta del portafolio en relación con el benchmark usando una regresión lineal.
        
        :param portfolio_returns: Retornos del portafolio.
        :return: Beta del portafolio.
        """
        # Usar los retornos del benchmark
        benchmark_returns = self.benchmark_prices['^GSPC'].values  # Supongamos que la columna es el S&P500
        
        # Ajustar el tamaño para que coincida si es necesario
        portfolio_returns = portfolio_returns.reshape(-1, 1)
        benchmark_returns = benchmark_returns.reshape(-1, 1)
        
        # Usamos una regresión lineal para estimar beta
        reg = LinearRegression().fit(benchmark_returns, portfolio_returns)
        beta = reg.coef_[0][0]
        
        return beta

    def calculate_jensen_alpha(self, weights):
        """
        Calcula el Jensen's Alpha del portafolio.
        
        :param weights: Pesos de los activos en el portafolio.
        :return: Jensen's Alpha.
        """
        # Retornos del portafolio con los pesos dados
        portfolio_returns = np.dot(self.asset_prices, weights)
        
        # Retornos promedio del portafolio y del benchmark
        portfolio_avg_return = np.mean(portfolio_returns)
        benchmark_avg_return = np.mean(self.benchmark_prices['^GSPC'])
        
        # Calcular beta del portafolio en relación con el benchmark
        beta = self.calculate_beta(portfolio_returns)
        
        # Aplicar la fórmula de Jensen's Alpha
        jensen_alpha = (portfolio_avg_return - self.rf) - beta * (benchmark_avg_return - self.rf)
        
        return jensen_alpha

    def calculate_sharpe_ratio(self, weights):
        """
        Calcula el Sharpe Ratio del portafolio.
        
        :param weights: Pesos de los activos en el portafolio.
        :return: Sharpe Ratio.
        """
        # Calcular los retornos esperados del portafolio
        portfolio_returns = np.dot(weights, self.asset_prices.mean())
        
        # Calcular la varianza del portafolio
        portfolio_variance = np.dot(weights.T, np.dot(self.asset_prices.cov(), weights))
        
        # Calcular el Sharpe Ratio
        sharpe_ratio = (portfolio_returns - self.rf) / np.sqrt(portfolio_variance)
        return sharpe_ratio


    def neg_omega_ratio(self, weights):
        """
        Calcula el Omega Ratio negativo.
        
        :param weights: Pesos de los activos en el portafolio.
        :return: Omega Ratio negativo.
        """
        # Calcular los retornos del portafolio con los pesos dados
        portfolio_returns = np.dot(self.asset_prices, weights)

        # Extraer los retornos del benchmark de la columna ^GSPC de benchmark_prices
        benchmark_returns = self.benchmark_prices['^GSPC'].values  # Convertir a NumPy array

        # Verificar si las longitudes de benchmark_returns y portfolio_returns coinciden
        if len(benchmark_returns) != len(portfolio_returns):
            raise ValueError(f"El número de retornos del benchmark ({len(benchmark_returns)}) debe coincidir con los retornos del portafolio ({len(portfolio_returns)}).")

        # Calcular los retornos en exceso
        excess_returns = portfolio_returns - benchmark_returns

        # Calcular el Omega Ratio
        positive_excess = excess_returns[excess_returns > 0].sum()
        negative_excess = -excess_returns[excess_returns < 0].sum()

        if negative_excess == 0:
            return np.inf

        omega_ratio = positive_excess / negative_excess
        return -omega_ratio

    def neg_sortino_ratio(self, weights):
        """  
        Calcula el Sortino Ratio negativo.
        
        :param weights: Pesos de los activos en el portafolio.
        :return: Sortino Ratio negativo.
        """
        portfolio_return = np.dot(weights, self.average_asset_prices)
        excess_returns = self.asset_prices - self.rf
        negative_excess_returns = excess_returns[excess_returns < 0]
        weighted_negative_excess_returns = negative_excess_returns.multiply(weights, axis=1)
        semivariance = np.mean(np.square(weighted_negative_excess_returns.sum(axis=1)))

        if semivariance == 0:
            return np.inf

        sortino_ratio = (portfolio_return - self.rf) / np.sqrt(semivariance)
        return -sortino_ratio

    def optimize_with_ranking(self, num_portfolios=1000, strategy='sharpe'):
        """
        Optimiza los portafolios y asigna un ranking basado en la estrategia especificada.

        :param num_portfolios: Número de portafolios a generar.
        :param strategy: Estrategia para la optimización ('sharpe', 'omega', 'sortino').
        :return: Lista de portafolios ordenados y sus ponderaciones inversas.
        """
        portfolio_stats = []
        
        for _ in range(num_portfolios):
            # Generar pesos aleatorios para los activos
            weights = np.random.random(len(self.asset_prices.columns))
            weights /= np.sum(weights)
            
            # Calcular la métrica basada en la estrategia elegida
            if strategy == 'sharpe':
                score = self.calculate_sharpe_ratio(weights)
            elif strategy == 'omega':
                score = self.neg_omega_ratio(weights)
            elif strategy == 'sortino':
                score = self.neg_sortino_ratio(weights)
            else:
                raise ValueError("Estrategia desconocida: usa 'sharpe', 'omega', o 'sortino'")
            
            portfolio_stats.append((weights, score))
        
        # Ordenar los portafolios en orden inverso (mejor primero)
        ranked_portfolios = sorted(portfolio_stats, key=lambda x: x[1], reverse=True)

        # Asignar ponderaciones inversas (el mejor obtiene mayor peso)
        total_portfolios = len(ranked_portfolios)
        inverse_weights = [(i+1)/total_portfolios for i in range(total_portfolios)]
        
        return ranked_portfolios, inverse_weights

    def optimize_with_multiple_strategies(self, num_portfolios=1000, strategies=['sharpe', 'omega', 'sortino']):
        """
        Optimiza los portafolios para múltiples estrategias y devuelve los resultados.

        :param num_portfolios: Número de portafolios a generar.
        :param strategies: Lista de estrategias a utilizar ('sharpe', 'omega', 'sortino', 'jensen').
        :return: Diccionario de portafolios óptimos para cada estrategia.
        """
        optimal_portfolios = {}

        # Iterar sobre cada estrategia
        for strategy in strategies:
            portfolio_stats = []
            
            for _ in range(num_portfolios):
                # Generar pesos aleatorios para los activos
                weights = np.random.random(len(self.asset_prices.columns))
                weights /= np.sum(weights)
                
                # Calcular la métrica basada en la estrategia elegida
                if strategy == 'sharpe':
                    score = self.calculate_sharpe_ratio(weights)
                elif strategy == 'omega':
                    score = self.neg_omega_ratio(weights)
                elif strategy == 'sortino':
                    score = self.neg_sortino_ratio(weights)
                else:
                    raise ValueError("Estrategia desconocida: usa 'sharpe', 'omega', 'sortino'")
                
                portfolio_stats.append((weights, score))
            
            # Ordenar los portafolios en orden inverso (mejor primero)
            ranked_portfolios = sorted(portfolio_stats, key=lambda x: x[1], reverse=True)
            
            # Guardar los mejores portafolios para esta estrategia
            optimal_portfolios[strategy] = ranked_portfolios

        return optimal_portfolios
