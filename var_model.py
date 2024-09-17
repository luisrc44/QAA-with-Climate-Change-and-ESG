class VAR(TimeSeriesModel):
    r"""
    Fit VAR(p) process and do lag order selection

    .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t

    Parameters
    ----------
    endog : array_like
        2-d endogenous response variable. The independent variable.
    exog : array_like
        2-d exogenous variable.
    dates : array_like
        must match number of rows of endog

    References
    ----------
    Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
    """

    y = deprecated_alias("y", "endog", remove_version="0.11.0")

    def __init__(
        self, endog, exog=None, dates=None, freq=None, missing="none"
    ):
        super().__init__(endog, exog, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError("Only gave one variable to VAR")
        self.neqs = self.endog.shape[1]
        self.n_totobs = len(endog)



    def predict(self, params, start=None, end=None, lags=1, trend="c"):
        """
        Returns in-sample predictions or forecasts
        """
        params = np.array(params)

        if start is None:
            start = lags

        # Handle start, end
        (
            start,
            end,
            out_of_sample,
            prediction_index,
        ) = self._get_prediction_index(start, end)

        if end < start:
            raise ValueError("end is before start")
        if end == start + out_of_sample:
            return np.array([])

        k_trend = util.get_trendorder(trend)
        k = self.neqs
        k_ar = lags

        predictedvalues = np.zeros((end + 1 - start + out_of_sample, k))
        if k_trend != 0:
            intercept = params[:k_trend]
            predictedvalues += intercept

        y = self.endog
        x = util.get_var_endog(y, lags, trend=trend, has_constant="raise")
        fittedvalues = np.dot(x, params)

        fv_start = start - k_ar
        pv_end = min(len(predictedvalues), len(fittedvalues) - fv_start)
        fv_end = min(len(fittedvalues), end - k_ar + 1)
        predictedvalues[:pv_end] = fittedvalues[fv_start:fv_end]

        if not out_of_sample:
            return predictedvalues

        # fit out of sample
        y = y[-k_ar:]
        coefs = params[k_trend:].reshape((k_ar, k, k)).swapaxes(1, 2)
        predictedvalues[pv_end:] = forecast(y, coefs, intercept, out_of_sample)
        return predictedvalues



    def fit(
        self,
        maxlags: int | None = None,
        method="ols",
        ic=None,
        trend="c",
        verbose=False,
    ):
        # todo: this code is only supporting deterministic terms as exog.
        # This means that all exog-variables have lag 0. If dealing with
        # different exogs is necessary, a `lags_exog`-parameter might make
        # sense (e.g. a sequence of ints specifying lags).
        # Alternatively, leading zeros for exog-variables with smaller number
        # of lags than the maximum number of exog-lags might work.
        """
        Fit the VAR model

        Parameters
        ----------
        maxlags : {int, None}, default None
            Maximum number of lags to check for order selection, defaults to
            12 * (nobs/100.)**(1./4), see select_order function
        method : {'ols'}
            Estimation method to use
        ic : {'aic', 'fpe', 'hqic', 'bic', None}
            Information criterion to use for VAR order selection.
            aic : Akaike
            fpe : Final prediction error
            hqic : Hannan-Quinn
            bic : Bayesian a.k.a. Schwarz
        verbose : bool, default False
            Print order selection output to the screen
        trend : str {"c", "ct", "ctt", "n"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "n" - co constant, no trend
            Note that these are prepended to the columns of the dataset.

        Returns
        -------
        VARResults
            Estimation results

        Notes
        -----
        See Lütkepohl pp. 146-153 for implementation details.
        """
        lags = maxlags
        if trend not in ["c", "ct", "ctt", "n"]:
            raise ValueError(f"trend '{trend}' not supported for VAR")

        if ic is not None:
            selections = self.select_order(maxlags=maxlags)
            if not hasattr(selections, ic):
                raise ValueError(
                    "%s not recognized, must be among %s"
                    % (ic, sorted(selections))
                )
            lags = getattr(selections, ic)
            if verbose:
                print(selections)
                print("Using %d based on %s criterion" % (lags, ic))
        else:
            if lags is None:
                lags = 1

        k_trend = util.get_trendorder(trend)
        orig_exog_names = self.exog_names
        self.exog_names = util.make_lag_names(self.endog_names, lags, k_trend)
        self.nobs = self.n_totobs - lags

        # add exog to data.xnames (necessary because the length of xnames also
        # determines the allowed size of VARResults.params)
        if self.exog is not None:
            if orig_exog_names:
                x_names_to_add = orig_exog_names
            else:
                x_names_to_add = [
                    ("exog%d" % i) for i in range(self.exog.shape[1])
                ]
            self.data.xnames = (
                self.data.xnames[:k_trend]
                + x_names_to_add
                + self.data.xnames[k_trend:]
            )
        self.data.cov_names = pd.MultiIndex.from_product(
            (self.data.xnames, self.data.ynames)
        )
        return self._estimate_var(lags, trend=trend)



    def _estimate_var(self, lags, offset=0, trend="c"):
        """
        lags : int
            Lags of the endogenous variable.
        offset : int
            Periods to drop from beginning-- for order selection so it's an
            apples-to-apples comparison
        trend : {str, None}
            As per above
        """
        # have to do this again because select_order does not call fit
        self.k_trend = k_trend = util.get_trendorder(trend)

        if offset < 0:  # pragma: no cover
            raise ValueError("offset must be >= 0")

        nobs = self.n_totobs - lags - offset
        endog = self.endog[offset:]
        exog = None if self.exog is None else self.exog[offset:]
        z = util.get_var_endog(endog, lags, trend=trend, has_constant="raise")
        if exog is not None:
            # TODO: currently only deterministic terms supported (exoglags==0)
            # and since exoglags==0, x will be an array of size 0.
            x = util.get_var_endog(
                exog[-nobs:], 0, trend="n", has_constant="raise"
            )
            x_inst = exog[-nobs:]
            x = np.column_stack((x, x_inst))
            del x_inst  # free memory
            temp_z = z
            z = np.empty((x.shape[0], x.shape[1] + z.shape[1]))
            z[:, : self.k_trend] = temp_z[:, : self.k_trend]
            z[:, self.k_trend : self.k_trend + x.shape[1]] = x
            z[:, self.k_trend + x.shape[1] :] = temp_z[:, self.k_trend :]
            del temp_z, x  # free memory
        # the following modification of z is necessary to get the same results
        # as JMulTi for the constant-term-parameter...
        for i in range(self.k_trend):
            if (np.diff(z[:, i]) == 1).all():  # modify the trend-column
                z[:, i] += lags
            # make the same adjustment for the quadratic term
            if (np.diff(np.sqrt(z[:, i])) == 1).all():
                z[:, i] = (np.sqrt(z[:, i]) + lags) ** 2

        y_sample = endog[lags:]
        # Lütkepohl p75, about 5x faster than stated formula
        params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]
        resid = y_sample - np.dot(z, params)

        # Unbiased estimate of covariance matrix $\Sigma_u$ of the white noise
        # process $u$
        # equivalent definition
        # .. math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1}
        # Z^\prime) Y
        # Ref: Lütkepohl p.75
        # df_resid right now is T - Kp - 1, which is a suggested correction

        avobs = len(y_sample)
        if exog is not None:
            k_trend += exog.shape[1]
        df_resid = avobs - (self.neqs * lags + k_trend)

        sse = np.dot(resid.T, resid)
        if df_resid:
            omega = sse / df_resid
        else:
            omega = np.full_like(sse, np.nan)

        varfit = VARResults(
            endog,
            z,
            params,
            omega,
            lags,
            names=self.endog_names,
            trend=trend,
            dates=self.data.dates,
            model=self,
            exog=self.exog,
        )
        return VARResultsWrapper(varfit)



    def select_order(self, maxlags=None, trend="c"):
        """
        Compute lag order selections based on each of the available information
        criteria

        Parameters
        ----------
        maxlags : int
            if None, defaults to 12 * (nobs/100.)**(1./4)
        trend : str {"n", "c", "ct", "ctt"}
            * "n" - no deterministic terms
            * "c" - constant term
            * "ct" - constant and linear term
            * "ctt" - constant, linear, and quadratic term

        Returns
        -------
        selections : LagOrderResults
        """
        ntrend = len(trend) if trend.startswith("c") else 0
        max_estimable = (self.n_totobs - self.neqs - ntrend) // (1 + self.neqs)
        if maxlags is None:
            maxlags = int(round(12 * (len(self.endog) / 100.0) ** (1 / 4.0)))
            # TODO: This expression shows up in a bunch of places, but
            #  in some it is `int` and in others `np.ceil`.  Also in some
            #  it multiplies by 4 instead of 12.  Let's put these all in
            #  one place and document when to use which variant.

            # Ensure enough obs to estimate model with maxlags
            maxlags = min(maxlags, max_estimable)
        else:
            if maxlags > max_estimable:
                raise ValueError(
                    "maxlags is too large for the number of observations and "
                    "the number of equations. The largest model cannot be "
                    "estimated."
                )

        ics = defaultdict(list)
        p_min = 0 if self.exog is not None or trend != "n" else 1
        for p in range(p_min, maxlags + 1):
            # exclude some periods to same amount of data used for each lag
            # order
            result = self._estimate_var(p, offset=maxlags - p, trend=trend)

            for k, v in result.info_criteria.items():
                ics[k].append(v)

        selected_orders = {
            k: np.array(v).argmin() + p_min for k, v in ics.items()
        }

        return LagOrderResults(ics, selected_orders, vecm=False)



    @classmethod
    def from_formula(
        cls, formula, data, subset=None, drop_cols=None, *args, **kwargs
    ):
        """
        Not implemented. Formulas are not supported for VAR models.
        """
        raise NotImplementedError("formulas are not supported for VAR models.")


