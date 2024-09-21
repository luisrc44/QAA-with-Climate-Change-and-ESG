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


class EconomicDataCleaner:
    """
    This class is responsible for loading, cleaning, and preparing economic data such as GDP and TB3.
    """
    
    def __init__(self, gdp_path, tb3_path, cpi_path):
        """
        Initialize the EconomicDataCleaner with file paths for the datasets.
        
        :param gdp_path: Path to the GDP CSV file.
        :param tb3_path: Path to the TB3 CSV file.
        """
        self.gdp_path = gdp_path
        self.tb3_path = tb3_path
        self.cpi_path = cpi_path

    def load_data(self):
        """
        Load the GDP and TB3 datasets.
        """
        self.gdp = pd.read_csv(self.gdp_path)
        self.tb3 = pd.read_csv(self.tb3_path)
        self.cpi = pd.read_csv(self.cpi_path)

    def clean_gdp(self):
        """
        Clean and process the GDP dataset.
        """
        self.gdp = self.gdp.rename(columns={'DATE': 'Date', 'GDP': 'GDP'})
        self.gdp['Date'] = pd.to_datetime(self.gdp['Date'])
        #self.gdp['GDP_pct_change'] = self.gdp['GDP'].pct_change(periods=12).dropna()

    def clean_tb3(self):
        """
        Clean and process the TB3 dataset.
        """
        self.tb3 = self.tb3.rename(columns={'DATE': 'Date', 'TB3MS': 'TB3MS'})
        self.tb3['Date'] = pd.to_datetime(self.tb3['Date'])

        # Interpolate missing values for monthly alignment
        #self.tb3 = self.tb3.set_index('Date').resample('M').interpolate(method='linear').reset_index()

    def clean_cpi(self):
        """
        Clean and process the GDP dataset.
        """
        self.cpi = self.cpi.rename(columns={'DATE': 'Date', 'CPIAUCSL': 'CPI'})
        self.cpi['Date'] = pd.to_datetime(self.cpi['Date'])
        #self.gdp['GDP_pct_change'] = self.gdp['GDP'].pct_change(periods=12).dropna()

    def merge_data(self):
        """
        Merge the cleaned GDP, TB3, and CPI datasets on the 'Date' column, resample to monthly frequency, and fill missing months with the value of the first month in the quarter.
        
        :return: Merged DataFrame with GDP, TB3, and CPI starting from 2014-12-01 with monthly data.
        """
        # Ensure all Date columns are in datetime format
        self.gdp['Date'] = pd.to_datetime(self.gdp['Date'], errors='coerce')
        self.tb3['Date'] = pd.to_datetime(self.tb3['Date'], errors='coerce')
        self.cpi['Date'] = pd.to_datetime(self.cpi['Date'], errors='coerce')

        # Merge GDP and TB3 first
        gdp_tb3 = pd.merge(self.gdp[['Date', 'GDP']], self.tb3[['Date', 'TB3MS']], on='Date', how='inner')
        
        # Then merge the result with CPI
        data = pd.merge(gdp_tb3, self.cpi[['Date', 'CPI']], on='Date', how='inner')

        # Set the date column as the index
        data.set_index('Date', inplace=True)

        # Resample to monthly frequency, using forward fill for GDP
        data = data.resample('M').ffill()

        # Convert the start date to datetime format
        start_date = pd.to_datetime('2014-12-01')

        # Filter the data to include only records from 2014-12-01 onward
        data = data[data.index >= start_date]

        # Set all dates to the first day of the month
        data.index = data.index.map(lambda x: x.replace(day=1))
        return data

    def clean_and_prepare_data(self):
        """
        Main function to load, clean, and merge all economic datasets.
        
        :return: Cleaned and merged economic dataset.
        """
        # Load datasets
        self.load_data()

        # Clean individual datasets
        self.clean_gdp()
        self.clean_tb3()
        self.clean_cpi()

        # Merge datasets and return the result
        return self.merge_data()


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
        Main function to load, clean, and merge all climate datasets.
        
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
