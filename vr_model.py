import optuna
import warnings
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.tools import add_constant

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(logging.ERROR)


# Clase ClimateVR
class ClimateVR:
    def __init__(self, endog, exog=None, climate_data=None, dates=None, freq=None, missing="none"):
        self.endog = endog
        self.exog = exog
        self.climate_data = climate_data
        self.dates = dates
        self.freq = freq
        self.missing = missing

        if self.endog.ndim == 1:
            raise ValueError("Solo se proporcionó una variable, el modelo VR necesita múltiples variables.")
        self.neqs = self.endog.shape[1]  # Número de ecuaciones (variables endógenas)
        self.n_totobs = len(endog)  # Número total de observaciones

        if self.climate_data is not None:
            if len(self.endog) != len(self.climate_data):
                raise ValueError("Los datos endógenos y climáticos deben tener la misma longitud.")
            self.endog = np.hstack((self.endog, self.climate_data))

    def fit(self, maxlags=None, method="ols", trend="c", verbose=False):
        if maxlags is None:
            maxlags = 1  # Se puede ajustar dinámicamente

        # Generar matriz de retardos
        z = self._get_lagged_endog(self.endog, maxlag=maxlags, trend=trend)
        params = np.linalg.lstsq(z, self.endog[maxlags:], rcond=None)[0]

        # Guardar las dimensiones de z y endog ajustados para la predicción
        self.maxlags = maxlags
        self.z_shape = z.shape

        return params

    def _get_lagged_endog(self, endog, maxlag, trend):
        z = lagmat(endog, maxlag=maxlag, trim="both")
        if trend == "c":
            z = add_constant(z, prepend=False)
        return z

    def predict(self, params, start=None, end=None, lags=1, trend="c"):
        # Verificar que end esté definido, si no, asumir que es el final de los datos disponibles
        if end is None:
            end = self.n_totobs - 1

        if start is None:
            start = lags

        # Definir el número de observaciones futuras a predecir
        num_predictions = end + 1 - start

        # Asegurarse de que el número de predicciones sea válido
        if num_predictions <= 0:
            raise ValueError("El rango de predicción es inválido. Asegúrate de que 'end' sea mayor que 'start'.")

        # Crear array vacío para valores predichos
        predictedvalues = np.zeros((num_predictions, self.neqs))

        # Obtener la matriz de retardos
        z = self._get_lagged_endog(self.endog, maxlag=lags, trend=trend)

        # Asegurarse de que las dimensiones de z y params sean compatibles
        z = z[:, :params.shape[0]]  # Alinear dimensiones truncando z si es necesario

        # Calcular los valores ajustados
        fittedvalues = np.dot(z, params)

        # Ajustar el número de valores predichos para que coincida con la longitud disponible
        if len(fittedvalues) > len(predictedvalues):
            fittedvalues = fittedvalues[:len(predictedvalues)]

        # Asignar los valores ajustados a predictedvalues
        predictedvalues[:len(fittedvalues)] = fittedvalues

        return predictedvalues

    def optimize_maxlags(self, all_data, n_trials=50):
        """
        Optimiza el número de retardos (maxlags) utilizando Optuna.

        :param all_data: Conjunto de datos para entrenar el modelo y realizar predicciones.
        :param n_trials: Número de pruebas que realizará Optuna para encontrar el mejor maxlags.
        :return: Número óptimo de retardos (maxlags) y RMSE correspondiente.
        """
        
        def objective(trial):
            maxlags = trial.suggest_int('maxlags', 1, 10) # Elige el número de retardos entre 1 y 10
            vr_results = self.fit(maxlags=maxlags) # Ajustar el modelo VR con el número de retardos sugerido
            predicted = self.predict(vr_results, lags=maxlags, end=len(all_data)) # Realizar predicciones con el número de lags
            actual = all_data[maxlags:len(all_data)]

            # Asegurarse de que predicted y actual tengan la misma longitud
            min_len = min(len(predicted), len(actual))
            predicted = predicted[:min_len]
            actual = actual[:min_len]

            # Calcular el error de predicción (RMSE)
            rmse = np.sqrt(mean_squared_error(actual, predicted))

            return rmse
        
        # Crear el estudio de Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_trial.params['maxlags'], study.best_value
    
    def simulate_scenarios(self, vr_results, lags, n_scenarios=100, periods=10, noise_std=0.01):
        """
        Simula escenarios futuros para las variables no financieras utilizando un random walk basado en el modelo VR.
        Se añade ruido gaussiano a las predicciones para generar escenarios múltiples.
        
        :param vr_results: Los parámetros ajustados del modelo VR.
        :param lags: El número de retardos (lags) utilizados en el modelo VR.
        :param n_scenarios: Número de escenarios a generar.
        :param periods: Número de periodos futuros a simular.
        :param noise_std: Desviación estándar del ruido gaussiano a añadir a las predicciones.
        :return: Diccionario de escenarios simulados para cada variable.
        """
        # Obtener las dimensiones de las variables (número de variables no financieras)
        n_variables = self.neqs  # Número de variables en el modelo VR
        scenarios = {}

        # Inicializar escenarios para cada variable
        for var_idx, var_name in enumerate(self.endog.columns):
            scenarios[var_name] = np.zeros((periods, n_scenarios))

        # Simulación de escenarios
        for scenario in range(n_scenarios):
            # Predecir los valores para los próximos 'periods' utilizando el modelo ajustado VR
            future_values = self.predict(vr_results, start=len(self.endog), end=len(self.endog) + periods - 1, lags=lags)
            
            # Añadir ruido gaussiano a las predicciones para generar trayectorias diferentes
            noise = np.random.normal(0, noise_std, future_values.shape)
            future_values_noisy = future_values + noise
            
            # Almacenar las predicciones con ruido para cada variable
            for var_idx, var_name in enumerate(self.endog.columns):
                scenarios[var_name][:, scenario] = future_values_noisy[:, var_idx]
        
        return scenarios
