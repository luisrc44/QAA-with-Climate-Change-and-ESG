import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, expected_returns, cov_matrix, risk_free_rate):
        """
        Inicializa la clase PortfolioOptimizer con los retornos esperados y la matriz de covarianzas.
        
        :param expected_returns: Retornos esperados de los activos.
        :param cov_matrix: Matriz de covarianzas de los retornos.
        :param risk_free_rate: Tasa libre de riesgo (rf).
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.rf = risk_free_rate
    
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
    
    def optimize_portfolio(self):
        """
        Optimiza el portafolio para maximizar el Sharpe Ratio.
        
        :return: Pesos óptimos del portafolio.
        """
        # Restricciones: la suma de los pesos debe ser 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Fronteras: pesos entre 0 y 1 (no vendemos activos a corto)
        bounds = tuple((0, 1) for _ in range(len(self.expected_returns)))
        
        # Peso inicial para la optimización (pesos igual distribuidos)
        initial_weights = len(self.expected_returns) * [1. / len(self.expected_returns)]
        
        # Optimización para maximizar el Sharpe Ratio
        opt_results = sco.minimize(self.negative_sharpe_ratio, initial_weights, 
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
        plt.xlabel('Riesgo (Volatilidad)')
        plt.ylabel('Retorno')
        plt.title('Frontera Eficiente')
        plt.show()
