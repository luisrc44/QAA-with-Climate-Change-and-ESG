import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, asset_prices, risk_free_rate, benchmark_prices=None, economic_factors=None, climate_factors=None, betas=None):
        self.asset_prices = asset_prices.select_dtypes(include=[np.number])
        self.rf = risk_free_rate
        self.benchmark_prices = benchmark_prices
        self.economic_factors = economic_factors
        self.climate_factors = climate_factors
        self.betas = betas  
        self.average_asset_prices = self.asset_prices


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

    def objective_function(self, weights, metric):
        if metric == 'sharpe':
            return -self.calculate_sharpe_ratio(weights)
        elif metric == 'omega':
            return self.neg_omega_ratio(weights)
        elif metric == 'sortino':
            return self.neg_sortino_ratio(weights)

    def optimize_with_slsqp(self, initial_weights, bounds, constraints, metric):
        result = minimize(self.objective_function, initial_weights, args=(metric,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def random_portfolio(self, num_assets, green_weight, gray_weight, num_green, num_gray):
        weights = np.random.random(num_assets)
        weights[:num_gray] *= gray_weight
        weights[num_gray:] *= green_weight
        return weights / np.sum(weights)

    def generate_portfolios(self, num_portfolios=1000, metric='sharpe', green_weight=0.5, gray_weight=0.5, num_green=0, num_gray=0):
        portfolio_stats = []
        for _ in range(num_portfolios):
            weights = self.random_portfolio(num_green + num_gray, green_weight, gray_weight, num_green, num_gray)
            if metric == 'sharpe':
                score = self.calculate_sharpe_ratio(weights)
            elif metric == 'omega':
                score = -self.neg_omega_ratio(weights)
            elif metric == 'sortino':
                score = -self.neg_sortino_ratio(weights)
            portfolio_stats.append((weights, score))
        return portfolio_stats

    def optimize_with_multiple_strategies(self, strategies=['sharpe', 'omega', 'sortino'],
                                          gray_assets=None, green_assets=None):
        optimal_portfolios = {}
        all_portfolios = []

        allocations = [
            (0.75, 0.25),  # 75% green, 25% gray
            (0.50, 0.50),  # 50% green, 50% gray
            (0.25, 0.75)   # 25% green, 75% gray
        ]

        for green_weight, gray_weight in allocations:
            for strategy in strategies:
                portfolio_stats = []

                num_green = len(green_assets)
                num_gray = len(gray_assets)
                total_assets = num_green + num_gray

                # Inicialización de pesos
                initial_weights = np.ones(total_assets) / total_assets

                # Bounds (con mínimo del 2% para SLSQP)
                bounds_green = [(0.02, green_weight) for _ in range(num_green)]
                bounds_gray = [(0.02, gray_weight) for _ in range(num_gray)]
                bounds = bounds_gray + bounds_green  # Primero gray, luego green

                # Restricciones
                constraints = [
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Sum of weights = 1
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights[:num_gray]) - gray_weight},  # Gray weight
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights[num_gray:]) - green_weight}  # Green weight
                ]

                # Primer portafolio optimizado con SLSQP
                result = self.optimize_with_slsqp(initial_weights, bounds, constraints, strategy)
                optimized_weights = result.x

                if strategy == 'sharpe':
                    score = self.calculate_sharpe_ratio(optimized_weights)
                elif strategy == 'omega':
                    score = -self.neg_omega_ratio(optimized_weights)
                elif strategy == 'sortino':
                    score = -self.neg_sortino_ratio(optimized_weights)

                portfolio_stats.append((optimized_weights, score))
                all_portfolios.append({
                    'weights': np.round(optimized_weights, 2),  # Redondear los pesos a 4 decimales
                    'metric_score': score,
                    'strategy': strategy,
                    'green_weight': green_weight,
                    'gray_weight': gray_weight
                })

                # Generar 1000 portafolios aleatorios y seleccionar los mejores 4
                random_portfolios = self.generate_portfolios(
                    num_portfolios=10000, metric=strategy,
                    green_weight=green_weight, gray_weight=gray_weight,
                    num_green=num_green, num_gray=num_gray
                )
                top_random_portfolios = sorted(random_portfolios, key=lambda x: x[1], reverse=True)[:4]

                for random_weights, random_score in top_random_portfolios:
                    portfolio_stats.append((random_weights, random_score))
                    all_portfolios.append({
                        'weights': np.round(random_weights, 2),  # Redondear los pesos a 4 decimales
                        'metric_score': random_score,
                        'strategy': strategy,
                        'green_weight': green_weight,
                        'gray_weight': gray_weight
                    })

                # Ordenar y seleccionar los mejores 5 portafolios
                sorted_portfolios = sorted(portfolio_stats, key=lambda x: x[1], reverse=True)
                key = f"{strategy}_{int(green_weight*100)}green_{int(gray_weight*100)}gray"
                optimal_portfolios[key] = sorted_portfolios[:5]

        # Convertir todos los portafolios en un DataFrame
        portfolios_df = pd.DataFrame(all_portfolios)

        return optimal_portfolios, portfolios_df

    # Función para calcular el retorno esperado de los portafolios
    def calculate_expected_return(self, weights, asset_prices):
        return np.dot(weights, asset_prices.mean())


    def calculate_adjusted_returns(self, var_predictions):
        """
        Ajusta los retornos esperados de los activos usando las betas y las predicciones del VAR.

        :param var_predictions: NumPy array con las predicciones del VAR para las variables económicas y climáticas.
        :return: DataFrame con los retornos ajustados de los activos.
        """
        if self.betas is None:
            raise ValueError("Las betas no han sido proporcionadas.")

        # Asegurarse de que las predicciones estén en formato DataFrame
        # Crear un DataFrame para las predicciones usando las mismas columnas que las betas y generar un índice temporal adecuado
        var_predictions_df = pd.DataFrame(var_predictions, columns=self.betas.columns)

        # Calcular los retornos ajustados
        adjusted_returns = np.dot(self.betas.values, var_predictions_df.T)

        return pd.DataFrame(adjusted_returns, index=self.betas.index, columns=var_predictions_df.index)

