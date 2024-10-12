import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PortfolioBacktester:
    def __init__(self, asset_returns, portfolio_weights, initial_investment=10000):
        """
        Inicializa el backtester del portafolio con los datos proporcionados.
        
        :param asset_returns: DataFrame con los retornos históricos de los activos.
        :param portfolio_weights: Pesos del mejor portafolio.
        :param initial_investment: Monto inicial de inversión (por defecto 10,000).
        """
        self.asset_returns = asset_returns
        self.portfolio_weights = portfolio_weights
        self.initial_investment = initial_investment
        self.backtest_df = None

    def run_backtest(self):
        """
        Realiza el backtesting del portafolio aplicando los pesos a los retornos históricos de los activos.
        
        :return: DataFrame con el valor del portafolio a lo largo del tiempo y métricas clave.
        """
        # Aplicar los pesos a los retornos de los activos para obtener los retornos del portafolio
        portfolio_returns = self.asset_returns.dot(self.portfolio_weights)

        # Calcular el valor acumulado del portafolio
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = self.initial_investment * cumulative_returns

        # Calcular drawdown máximo
        running_max = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calcular el Sharpe ratio (ajustado a la tasa libre de riesgo)
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)  # 252 días de trading en un año

        # Crear un DataFrame para almacenar los resultados
        self.backtest_df = pd.DataFrame({
            'Portfolio Value': portfolio_value,
            'Portfolio Returns': portfolio_returns,
            'Cumulative Returns': cumulative_returns,
            'Drawdown': drawdown
        })

        # Retornar métricas clave
        return sharpe_ratio, max_drawdown

    def plot_results(self):
        """
        Grafica el valor del portafolio a lo largo del tiempo y el drawdown.
        """
        if self.backtest_df is None:
            raise ValueError("Backtest no ha sido ejecutado. Llama a run_backtest() antes de graficar.")
        
        plt.figure(figsize=(12, 8))
        
        # Graficar el valor del portafolio
        plt.subplot(2, 1, 1)
        plt.plot(self.backtest_df.index, self.backtest_df['Portfolio Value'], label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        
        # Graficar el drawdown
        plt.subplot(2, 1, 2)
        plt.plot(self.backtest_df.index, self.backtest_df['Drawdown'], label='Drawdown', color='red')
        plt.title('Portfolio Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

