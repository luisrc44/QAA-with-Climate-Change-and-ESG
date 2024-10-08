import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import optuna
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging
import warnings

# Configurar el nivel de logging de optuna a ERROR
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Suprimir las advertencias de statsmodels y pandas
warnings.filterwarnings("ignore", category=UserWarning)


class VARModel:
    def __init__(self, data):
        """
        Inicializa la clase con los datos que serán usados para el VAR.
        Los datos se escalan para estabilizar las magnitudes.
        """
        self.data = data.set_index('Date')  # Asegura que las fechas sean el índice
        self.scaler = StandardScaler()
        self.data_scaled = pd.DataFrame(self.scaler.fit_transform(self.data), index=self.data.index, columns=self.data.columns)
        self.model = None
        self.lag_order = None

    def fit_var(self, lags):
        """
        Ajusta el modelo VAR con un número de lags especificado.
        En caso de error con la matriz no positiva definida, retorna np.inf para que Optuna lo descarte.
        """
        try:
            model = VAR(self.data_scaled)
            self.model = model.fit(lags)
            return self.model.aic  # Retorna el AIC para usarlo en Optuna
        except np.linalg.LinAlgError:
            return np.inf  # Retornar un valor alto para que Optuna lo descarte
        except ValueError:
            return np.inf  # Captura errores adicionales de los datos

    def predict(self, steps=3):
        """
        Realiza las predicciones para los próximos pasos (steps) y desescala las predicciones.
        """
        forecast_scaled = self.model.forecast(self.model.endog[-self.lag_order:], steps=steps)
        forecast = self.scaler.inverse_transform(forecast_scaled)
        return forecast

    def optimize_lags(self, trials=100):
        """
        Optimiza el número de lags usando Optuna. Se ha reducido el rango de sugerencias de lags.
        """
        def objective(trial):
            lags = trial.suggest_int('lags', 1, 5)  # Reducir el rango a 1-5 lags
            aic = self.fit_var(lags)
            return aic
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=trials)
        
        self.lag_order = study.best_params['lags']
        print(f"Optimal number of lags: {self.lag_order}")

    def plot_predictions(self, steps=3):
        """
        Grafica las predicciones futuras y las series originales.
        """
        forecast = self.predict(steps)
        
        # Corregir el índice para las predicciones basado en las fechas reales del dataframe
        last_date = self.data.index[-1]
        forecast_index = pd.date_range(last_date, periods=steps+1, freq='MS')[1:]  # 'MS' para que tome el inicio de cada mes

        # Graficar todas las variables
        for i, col in enumerate(self.data.columns):
            plt.figure(figsize=(10, 6))
            plt.plot(self.data.index, self.data[col], label="Historical")
            plt.plot(forecast_index, forecast[:, i], label="Forecast", linestyle='--')
            plt.title(f"Forecast vs Historical: {col}")
            plt.legend()
            plt.show()
