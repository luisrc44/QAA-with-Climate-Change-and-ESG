import yfinance as yf
import pandas as pd
import pandas_datareader.data as web

class DataDownloader:
    
    def __init__(self):
        """
        Initializes the AssetDataDownloader object. This class is designed to download
        adjusted closing prices for specified financial assets and a benchmark over a given period.
        """
        pass

    def download_data(self, start_date: str, end_date: str, assets: list, benchmark: str) -> tuple:
        """
        Downloads adjusted closing prices for a list of assets and a benchmark
        for the specified period.

        :param start_date: Start date for data download, in 'YYYY-MM-DD' format.
        :param end_date: End date for data download, in 'YYYY-MM-DD' format.
        :param assets: List of strings with the tickers of the assets to download.
        :param benchmark: String with the ticker of the benchmark to download.
        :return: A tuple containing two DataFrames: the first with the adjusted closing prices
                 of the assets and the second with the adjusted closing prices of the benchmark.
        """
        
        # Download adjusted closing prices for the assets and the given benchmark
        asset_data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
        benchmark_data = pd.DataFrame(yf.download(benchmark, start=start_date, end=end_date)['Adj Close'])
        benchmark_data = benchmark_data.rename(columns={'Adj Close': benchmark})

        return pd.DataFrame(asset_data), benchmark_data
