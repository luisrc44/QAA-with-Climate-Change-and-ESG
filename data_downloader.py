import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class DataDownloader:
    
    def __init__(self):
        """
        Initializes the DataDownloader object. This class is designed to download
        adjusted closing prices for specified financial assets and a benchmark over a given period,
        and calculate their monthly returns.
        """
        pass

    def download_data(self, start_date: str, end_date: str, assets: list, benchmark: str) -> tuple:
        """
        Downloads adjusted closing prices for a list of assets and a benchmark
        for the specified period, and calculates their monthly returns.

        :param start_date: Start date for data download, in 'YYYY-MM-DD' format.
        :param end_date: End date for data download, in 'YYYY-MM-DD' format.
        :param assets: List of strings with the tickers of the assets to download.
        :param benchmark: String with the ticker of the benchmark to download.
        :return: A tuple containing two DataFrames: the first with the monthly returns
                 of the assets and the second with the monthly returns of the benchmark.
        """
        asset_data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
        benchmark_data = pd.DataFrame(yf.download(benchmark, start=start_date, end=end_date)['Adj Close'])
        benchmark_data.columns = [benchmark]

        def calculate_monthly_returns(df):
            return df.resample('M').last().pct_change().dropna().reset_index().assign(Date=lambda x: x['Date'].apply(lambda y: y.replace(day=1)))
        
        return calculate_monthly_returns(asset_data), calculate_monthly_returns(benchmark_data)


class BetaCalculator:
    def __init__(self, assets_returns: pd.DataFrame, benchmark_returns: pd.DataFrame, climate_data: pd.DataFrame):
        """
        Initializes the BetaCalculator object.

        :param assets_returns: DataFrame with monthly returns of financial assets.
        :param benchmark_returns: DataFrame with monthly returns of the benchmark.
        :param climate_data: DataFrame with monthly changes of climate variables.
        """
        self.assets_returns = assets_returns
        self.benchmark_returns = benchmark_returns
        self.climate_data = climate_data

    def calculate_betas(self) -> pd.DataFrame:
        """
        Calculates the beta values of the assets with respect to the benchmark and climate variables.
        
        :return: A DataFrame containing the beta values of each asset for each variable.
        """
        betas = {}

        for asset in self.assets_returns.columns[1:]:  # Exclude Date column
            asset_betas = {}

            # Combine benchmark returns and climate data with asset returns
            merged_data = pd.merge(self.assets_returns[['Date', asset]], self.benchmark_returns, on='Date', how='inner')
            merged_data = pd.merge(merged_data, self.climate_data, on='Date', how='inner')

            # Run regression for each variable (benchmark and climate variables)
            for variable in merged_data.columns[2:]:  # Start from 2 to skip Date and asset column
                X = merged_data[[variable]]
                y = merged_data[asset]

                model = LinearRegression().fit(X, y)
                beta = model.coef_[0]

                asset_betas[variable] = beta

            betas[asset] = asset_betas

        return pd.DataFrame(betas)


