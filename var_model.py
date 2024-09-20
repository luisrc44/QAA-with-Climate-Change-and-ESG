from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.tools import add_constant
import numpy as np

class ClimateVAR:
    def __init__(self, endog, exog=None, climate_data=None, dates=None, freq=None, missing="none"):
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

        if self.climate_data is not None:
            if len(self.endog) != len(self.climate_data):
                raise ValueError("Los datos endógenos y climáticos deben tener la misma longitud.")
            self.endog = np.hstack((self.endog, self.climate_data))

    def fit(self, maxlags=None, method="ols", trend="c", verbose=False):
        if maxlags is None:
            maxlags = 1  # Se puede ajustar dinámicamente

        # Aquí se cambia de 'lags' a 'maxlag'
        z = self._get_lagged_endog(self.endog, maxlag=maxlags, trend=trend)

        params = np.linalg.lstsq(z, self.endog[maxlags:], rcond=None)[0]
        return params

    def _get_lagged_endog(self, endog, maxlag, trend):
        # Utiliza 'maxlag' en lugar de 'lags'
        z = lagmat(endog, maxlag=maxlag, trim="both")
        if trend == "c":
            z = add_constant(z, prepend=False)
        return z

    def predict(self, params, start=None, end=None, lags=1, trend="c"):
        if start is None:
            start = lags
        predictedvalues = np.zeros((end + 1 - start, self.neqs))
        z = self._get_lagged_endog(self.endog, maxlag=lags, trend=trend)
        fittedvalues = np.dot(z, params)
        predictedvalues[:len(fittedvalues)] = fittedvalues
        return predictedvalues
