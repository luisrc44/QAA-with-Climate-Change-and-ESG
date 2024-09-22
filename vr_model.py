import numpy as np
import warnings
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.tools import add_constant
import optuna
from sklearn.metrics import mean_squared_error
import pandas as pd

# Ignorar advertencias
warnings.filterwarnings('ignore')

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

        # Ajustar el modelo VR
        params = np.linalg.lstsq(z, self.endog[maxlags:], rcond=None)[0]

        # Guardar las dimensiones de z y endog ajustados para la predicción
        self.maxlags = maxlags
        self.z_shape = z.shape

        return params

    def _get_lagged_endog(self, endog, maxlag, trend):
        # Utiliza 'maxlag' en lugar de 'lags'
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
