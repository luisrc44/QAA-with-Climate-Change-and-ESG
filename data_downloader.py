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


class ClimateDataCleaner:
    """
    This class is responsible for loading, cleaning, and preparing climate and CO2 emissions data. 
    It includes functionality for renaming columns, converting dates, calculating percentage changes, 
    and merging the datasets for temperature, drought, and CO2 emissions.
    """

    def __init__(self, temperature_path, drought_path, co2_path):
        """
        Initialize the ClimateDataCleaner with file paths for the datasets.
        
        :param temperature_path: Path to the temperature CSV file.
        :param drought_path: Path to the drought CSV file.
        :param co2_path: Path to the CO2 emissions Excel file.
        """
        self.temperature_path = temperature_path
        self.drought_path = drought_path
        self.co2_path = co2_path

    def load_data(self):
        """
        Load the temperature, drought, and CO2 emissions datasets.
        """
        self.temperature = pd.read_csv(self.temperature_path)
        self.drought = pd.read_csv(self.drought_path)
        
        # Load CO2 emissions data from Excel
        self.co2_emission = pd.read_excel(self.co2_path, skiprows=10).drop(index=0)
        self.co2_emission = self.co2_emission[["Month", 
                                               "Coal, Including Coal Coke Net Imports, CO2 Emissions",
                                               "Natural Gas, Excluding Supplemental Gaseous Fuels, CO2 Emissions", 
                                               "Petroleum, Excluding Biofuels, CO2 Emissions", 
                                               "Total Energy CO2 Emissions"]]

    def clean_co2_emission(self):
        """
        Clean and process the CO2 emissions dataset.
        """
        # Rename columns for clarity
        self.co2_emission = self.co2_emission.rename(columns={
            "Month": "Date",
            "Coal, Including Coal Coke Net Imports, CO2 Emissions": "Coal",
            "Natural Gas, Excluding Supplemental Gaseous Fuels, CO2 Emissions": "Natural Gas",
            "Petroleum, Excluding Biofuels, CO2 Emissions": "Petroleum",
            "Total Energy CO2 Emissions": "Total CO2 Emissions"
        })

        # Convert 'Date' to datetime format
        self.co2_emission['Date'] = pd.to_datetime(self.co2_emission['Date'])

        # Calculate year-to-year percentage change
        self.co2_emission_pct_change = self.co2_emission.set_index('Date').pct_change(periods=12).dropna().reset_index()

    def clean_temperature(self):
        """
        Clean and process the temperature dataset.
        """
        # Drop unnecessary columns and rename columns for clarity
        self.temperature = self.temperature.drop(columns=['Average surface temperature.1', 'Code', 'Entity', 'year'])
        self.temperature = self.temperature.rename(columns={'Day': 'Date', 'Average surface temperature': 'Temperature'})
        
        # Convert 'Date' to datetime format
        self.temperature['Date'] = pd.to_datetime(self.temperature['Date'], format='%d/%m/%y', errors='coerce')
        self.temperature['Date'] = self.temperature['Date'].apply(lambda x: x.replace(year=x.year - 100) if x.year >= 2025 else x)
        
        # Filter dates between 1940 and 2024, and adjust day to 1
        self.temperature = self.temperature[(self.temperature['Date'] >= '1940-01-01') & (self.temperature['Date'] <= '2024-12-31')]
        self.temperature['Date'] = self.temperature['Date'].apply(lambda x: x.replace(day=1))
        
        # Calculate year-to-year percentage change
        temperature_pct = self.temperature.drop(columns=['Date']).pct_change(periods=12).dropna()
        temperature_pct['Date'] = self.temperature['Date']
        cols = ['Date'] + [col for col in temperature_pct.columns if col != 'Date']

        self.temperature = temperature_pct[cols]

    def clean_drought(self):
        """
        Clean and process the drought dataset.
        """
        # Rename columns for clarity and convert 'Date' to datetime format
        self.drought = self.drought.rename(columns={'MapDate': 'Date'})
        self.drought['Date'] = pd.to_datetime(self.drought['Date'], format='%Y%m%d')
        
        # Ensure DSCI is numeric and remove NaNs
        self.drought['DSCI'] = pd.to_numeric(self.drought['DSCI'], errors='coerce')

        # Group by year and month, then calculate the average DSCI
        self.drought['Year'] = self.drought['Date'].dt.year
        self.drought['Month'] = self.drought['Date'].dt.month
        monthly_avg = self.drought.groupby(['Year', 'Month'])['DSCI'].mean().reset_index()
        monthly_avg['Date'] = pd.to_datetime(monthly_avg.apply(lambda row: f"{int(row['Year'])}-{int(row['Month']):02d}-01", axis=1))
        
        # Keep only Date and DSCI columns
        self.drought = monthly_avg[['Date', 'DSCI']]

        # Calculate year-to-year percentage change
        #drought_pct = self.drought.drop(columns=['Date']).pct_change(periods=12).dropna()
        #drought_pct['Date'] = self.drought['Date']
        #self.drought = drought_pct

    def merge_data(self):
        """
        Merge the cleaned temperature, drought, and CO2 emission datasets on the 'Date' column.
        
        :return: Merged DataFrame with all datasets combined.
        """
        # Ensure 'Date' is in datetime format for all datasets
        self.temperature['Date'] = pd.to_datetime(self.temperature['Date'], errors='coerce')
        self.drought['Date'] = pd.to_datetime(self.drought['Date'], errors='coerce')
        self.co2_emission_pct_change['Date'] = pd.to_datetime(self.co2_emission_pct_change['Date'], errors='coerce')

        # Merge the datasets on the 'Date' column
        data = pd.merge(self.temperature, self.drought, on='Date', how='inner')
        data = pd.merge(data, self.co2_emission_pct_change, on='Date', how='inner')
        
        return data

    def clean_and_prepare_data(self):
        """
        Main function to load, clean, and merge all datasets.
        
        :return: Cleaned and merged dataset.
        """
        # Load datasets
        self.load_data()

        # Clean individual datasets
        self.clean_co2_emission()
        self.clean_temperature()
        self.clean_drought()

        # Merge datasets and return the result
        return self.merge_data()

