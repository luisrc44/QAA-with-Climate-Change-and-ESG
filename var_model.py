import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.tools import add_constant

class ClimateVAR:
    def __init__(self, endog, exog=None, climate_data=None, dates=None, freq=None, missing="none"):
        """
        Inicializa el modelo ClimateVAR, extendiendo VAR para incluir datos climáticos.

        :param endog: Variables endógenas (retornos financieros).
        :param exog: Variables macroeconómicas exógenas.
        :param climate_data: DataFrame con variables climáticas (e.g., cambio de temperatura).
        :param dates: Fechas asociadas con los datos.
        :param freq: Frecuencia de los datos de series temporales.
        :param missing: Cómo manejar datos faltantes.
        """
        self.endog = endog
        self.exog = exog
        self.climate_data = climate_data
        self.dates = dates
        self.freq = freq
        self.missing = missing
        
        if self.endog.ndim == 1:
            raise ValueError("Solo se proporcionó una variable, el modelo VAR necesita múltiples variables.")
        self.neqs = self.endog.shape[1]
        self.n_totobs = len(endog)

        # Combina las variables endógenas con los datos climáticos
        if self.climate_data is not None:
            if len(self.endog) != len(self.climate_data):
                raise ValueError("Los datos endógenos y climáticos deben tener la misma longitud.")
            # Añade los datos climáticos como variable exógena
            self.endog = np.hstack((self.endog, self.climate_data))

    def fit(self, maxlags=None, method="ols", trend="c", verbose=False):
        """
        Ajusta el modelo ClimateVAR.

        :param maxlags: Número máximo de retardos.
        :param method: Método de estimación (por ahora solo OLS).
        :param trend: Tipo de tendencia (c: constante, ct: constante y tendencia).
        :param verbose: Mostrar información adicional.
        :return: Estimaciones de los coeficientes.
        """
        if maxlags is None:
            maxlags = 1  # Se puede ajustar dinámicamente según los datos

        # Obtiene las variables endógenas con retardos
        z = self._get_lagged_endog(self.endog, maxlags, trend)

        # Realiza la estimación de los coeficientes usando OLS
        params = np.linalg.lstsq(z, self.endog[maxlags:], rcond=None)[0]
        return params

    def _get_lagged_endog(self, endog, lags, trend):
        """
        Calcula las variables endógenas con retardos.

        :param endog: Matriz de variables endógenas.
        :param lags: Número de retardos.
        :param trend: Tipo de tendencia.
        :return: Matriz de variables endógenas con retardos.
        """
        # Utiliza lagmat para crear retardos
        z = lagmat(endog, maxlags=lags, trim="both")
        if trend == "c":
            z = add_constant(z, prepend=False)
        return z

    def predict(self, params, start=None, end=None, lags=1, trend="c"):
        """
        Realiza predicciones in-sample o out-of-sample.

        :param params: Coeficientes del modelo ajustado.
        :param start: Inicio de la predicción.
        :param end: Fin de la predicción.
        :param lags: Número de retardos.
        :param trend: Tipo de tendencia.
        :return: Valores predichos.
        """
        if start is None:
            start = lags

        # Inicializa la matriz de valores predichos
        predictedvalues = np.zeros((end + 1 - start, self.neqs))

        # Obtener las variables endógenas con retardos
        z = self._get_lagged_endog(self.endog, lags, trend)

        # Realiza la predicción multiplicando por los parámetros estimados
        fittedvalues = np.dot(z, params)
        predictedvalues[:len(fittedvalues)] = fittedvalues
        return predictedvalues

    def predict_with_climate_scenarios(self, params, climate_scenario, start=None, end=None, lags=1, trend="c"):
        """
        Realiza predicciones con escenarios climáticos.

        :param params: Coeficientes del modelo ajustado.
        :param climate_scenario: Datos climáticos bajo diferentes escenarios.
        :param start: Inicio de la predicción.
        :param end: Fin de la predicción.
        :param lags: Número de retardos.
        :param trend: Tipo de tendencia.
        :return: Valores predichos basados en el escenario climático.
        """
        # Reemplaza los datos climáticos con los datos del escenario
        if len(climate_scenario) != (end - start):
            raise ValueError("La longitud del escenario climático debe coincidir con el período de predicción.")
        self.climate_data = climate_scenario
        return self.predict(params=params, start=start, end=end, lags=lags, trend=trend)
