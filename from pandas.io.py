from pandas.io.data import DataReader
from datetime import datetime

apple= DataReader('AAPL',  'yahoo', datetime(2010, 2, 1), datetime(2024, 11, 10))
print(apple['Adj Close'])