import pandas as pd

class DataProcessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def read_csv(self, file_path):
        self.dataframe = pd.read_csv(file_path)
        self.dataframe = self.dataframe.filter(["open", "high", "low", "close", "tick_volume"])
        self.dataframe = self.dataframe.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume"
        })

    def get_dataframe(self):
        return self.dataframe
    
    def get_close_prices(self):
        return self.dataframe["Close"].values
