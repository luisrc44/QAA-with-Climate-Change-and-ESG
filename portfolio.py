import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, expected_returns, cov_matrix, risk_free_rate, target_return=0.02):
        """
        Inicializa la clase PortfolioOptimizer con los retornos esperados y la matriz de covarianzas.
        
        :param expected_returns: Retornos esperados de los activos (acciones).
        :param cov_matrix: Matriz de covarianzas de los activos (acciones).
        :param risk_free_rate: Tasa libre de riesgo (rf).
        :param target_return: Retorno objetivo para el Omega Ratio.
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.rf = risk_free_rate
        self.target_return = target_return  # Para el Omega Ratio
    
    def portfolio_return(self, weights):
        """
        Calcula el retorno esperado de un portafolio.
        
        :param weights: Pesos del portafolio.
        :return: Retorno esperado del portafolio.
        """
        return np.dot(weights, self.expected_returns)
    
    def portfolio_variance(self, weights):
        """
        Calcula la varianza de un portafolio.
        
        :param weights: Pesos del portafolio.
        :return: Varianza del portafolio.
        """
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def negative_sharpe_ratio(self, weights):
        """
        Función objetivo para minimizar (Sharpe Ratio negativo).
        
        :param weights: Pesos del portafolio.
        :return: Sharpe Ratio negativo.
        """
        portfolio_ret = self.portfolio_return(weights)
        portfolio_vol = np.sqrt(self.portfolio_variance(weights))
        sharpe_ratio = (portfolio_ret - self.rf) / portfolio_vol
        return -sharpe_ratio
    
    def negative_sortino_ratio(self, weights):
        """
        Función objetivo para minimizar el Sortino Ratio negativo.
        Solo penaliza la desviación estándar negativa.
        
        :param weights: Pesos del portafolio.
        :return: Sortino Ratio negativo.
        """
        portfolio_ret = self.portfolio_return(weights)
        downside_risk = np.sqrt(np.mean(np.minimum(0, self.expected_returns - self.rf)**2))
        sortino_ratio = (portfolio_ret - self.rf) / downside_risk
        return -sortino_ratio
    
    def negative_omega_ratio(self, weights):
        """
        Función objetivo para minimizar el Omega Ratio negativo.
        Omega Ratio mide el ratio entre la probabilidad de superar el umbral de retorno objetivo 
        y la probabilidad de obtener retornos por debajo de ese umbral.
        
        :param weights: Pesos del portafolio.
        :return: Omega Ratio negativo.
        """
        portfolio_ret = self.portfolio_return(weights)
        excess_returns = self.expected_returns - self.target_return
        gains = np.sum(np.maximum(0, excess_returns))
        losses = -np.sum(np.minimum(0, excess_returns))
        omega_ratio = gains / losses
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
        bounds = tuple((0, 1) for _ in range(len(self.expected_returns)))
        
        # Peso inicial para la optimización (pesos igual distribuidos)
        initial_weights = len(self.expected_returns) * [1. / len(self.expected_returns)]
        
        # Selección de la estrategia
        if strategy == "sharpe":
            objective_function = self.negative_sharpe_ratio
        elif strategy == "sortino":
            objective_function = self.negative_sortino_ratio
        elif strategy == "omega":
            objective_function = self.negative_omega_ratio
        else:
            raise ValueError("Estrategia no reconocida. Usa 'sharpe', 'sortino' o 'omega'.")
        
        # Optimización
        opt_results = sco.minimize(objective_function, initial_weights, 
                                   method='SLSQP', bounds=bounds, constraints=constraints)
        
        return opt_results.x
    
    def simulate_random_portfolios(self, num_portfolios=5000):
        """
        Simula varios portafolios aleatorios para comparar la frontera eficiente.
        
        :param num_portfolios: Número de portafolios a simular.
        :return: Resultados simulados (retorno, volatilidad, Sharpe ratio).
        """
        results = np.zeros((3, num_portfolios))
        
        for i in range(num_portfolios):
            weights = np.random.random(len(self.expected_returns))
            weights /= np.sum(weights)
            
            portfolio_ret = self.portfolio_return(weights)
            portfolio_vol = np.sqrt(self.portfolio_variance(weights))
            sharpe_ratio = (portfolio_ret - self.rf) / portfolio_vol
            
            results[0, i] = portfolio_ret
            results[1, i] = portfolio_vol
            results[2, i] = sharpe_ratio
        
        return results
    
    def plot_efficient_frontier(self, num_portfolios=5000):
        """
        Grafica la frontera eficiente simulando portafolios aleatorios.
        
        :param num_portfolios: Número de portafolios a simular.
        """
        # Simular portafolios aleatorios
        results = self.simulate_random_portfolios(num_portfolios)
        
        # Graficar la frontera eficiente
        plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Risk (Volatility)')
        plt.ylabel('Return')
        plt.title('Efficient Frontier')
        plt.show()

