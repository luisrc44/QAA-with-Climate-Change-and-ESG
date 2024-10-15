from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import warnings
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)



class VR_Model:
    def __init__(self, data, lags):
        """
        Inicializa el VAR manual con datos y el número de lags.
        Los datos se escalan para estabilizar las magnitudes y se asegura que 'Date' no sea parte del conjunto de datos escalados.
        """
        self.original_data = data
        self.data = data.drop(columns=['Date'])
        self.lags = lags
        self.scaler = StandardScaler()
        self.data_scaled = pd.DataFrame(self.scaler.fit_transform(self.data), index=self.data.index, columns=self.data.columns)
        self.coefs = None

    def create_lagged_matrix(self):
        """
        Crea las matrices de diseño (X) y respuesta (Y) con lags.
        """
        X, Y = [], []
        for i in range(self.lags, len(self.data_scaled)):
            X.append(self.data_scaled.iloc[i-self.lags:i].values.flatten())
            Y.append(self.data_scaled.iloc[i].values)
        return np.array(X), np.array(Y)

    def fit(self):
        """
        Ajusta el modelo VAR usando regresión ordinaria.
        """
        X, Y = self.create_lagged_matrix()
        self.coefs = inv(X.T @ X) @ X.T @ Y

    def predict(self, steps=3):
        """
        Realiza predicciones a partir de los coeficientes ajustados y desescala las predicciones.
        """
        predictions = []
        last_values = self.data_scaled.values[-self.lags:]

        for _ in range(steps):
            X_pred = last_values.flatten()
            forecast = X_pred @ self.coefs
            predictions.append(forecast)
            
            last_values = np.vstack([last_values[1:], forecast])

        predictions = np.array(predictions)
        return self.scaler.inverse_transform(predictions)

    def plot_predictions(self, steps=3):
        """
        Grafica las predicciones futuras y las series originales, utilizando el índice de fechas del dataframe original.
        """
        forecast = self.predict(steps)
        
        last_date = self.original_data['Date'].iloc[-1]
        forecast_index = pd.date_range(last_date, periods=steps+1, freq='MS')[1:]

        for i, col in enumerate(self.data.columns):
            plt.figure(figsize=(10, 6))
            plt.plot(self.original_data['Date'], self.original_data[col], label="Historical")
            plt.plot(forecast_index, forecast[:, i], label="Forecast", linestyle='--')
            plt.title(f"Forecast vs Historical: {col}")
            plt.legend()
            plt.show()

    def generate_scenarios_with_noise(self, steps=3, n_scenarios=100):
        """
        Genera múltiples escenarios agregando ruido gaussiano a las predicciones.
        """
        forecast = self.predict(steps)
        scenarios = []

        for _ in range(n_scenarios):
            noise = np.random.normal(0, 0.04, forecast.shape)
            scenario = forecast + noise
            scenarios.append(scenario)

        scenarios = np.array(scenarios)

        last_date = self.original_data['Date'].iloc[-1]
        forecast_index = pd.date_range(last_date, periods=steps+1, freq='MS')[1:]

        for i, col in enumerate(self.data.columns):
            plt.figure(figsize=(10, 6))
            plt.plot(self.original_data['Date'], self.original_data[col], label="Historical")
            plt.plot(forecast_index, forecast[:, i], label="Forecast", linestyle='--', color='black')

            for scenario in scenarios:
                plt.plot(forecast_index, scenario[:, i], alpha=0.2)

            plt.title(f"Forecast with Noise Scenarios: {col}")
            plt.legend()
            plt.show()

def optimize_lags(data, trials=100):
    """
    Optimiza el número de lags usando Optuna para el modelo manual VAR.
    """
    def objective(trial):
        lags = trial.suggest_int('lags', 1, 5)
        var_model = VR_Model(data, lags)
        var_model.fit()
        X, Y = var_model.create_lagged_matrix()
        predictions = X @ var_model.coefs
        mse = np.mean((Y - predictions) ** 2)
        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)
    
    return study.best_params['lags']

def run_var_model(e_eco_data, steps=3, trials=50):
    """
    Optimiza el número de lags para el modelo VAR, ajusta el modelo y realiza predicciones.
    
    :param e_eco_data: DataFrame con los datos económicos y climáticos
    :param steps: Número de periodos a predecir
    :param trials: Número de pruebas para optimizar lags
    :return: Predicciones del modelo VAR
    """
    optimal_lags = optimize_lags(e_eco_data, trials=trials)
    vr_model = VR_Model(e_eco_data, optimal_lags)
    vr_model.fit()
    var_predictions = vr_model.predict(steps=steps)
    return var_predictions
