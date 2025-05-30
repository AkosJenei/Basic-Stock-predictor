# data_processing.py

import pandas as pd
from quantization import Quantization

class DataProcessor:
    def __init__(self):
        self.dataframe = None
        self.quantizer = None

    def read_csv(self, file_path: str, n_points: int, offset: int = 0):
        """
        Load exactly `n_points` rows starting at `offset` from the top of the CSV
        into self.dataframe, with proper renaming/filtering.
        """
        df = pd.read_csv(file_path)
        df = (
            df
            .filter(["open","high","low","close","tick_volume"])
            .rename(columns={
                "open":        "Open",
                "high":        "High",
                "low":         "Low",
                "close":       "Close",
                "tick_volume": "Volume"
            })
        )
        # slice rows [offset : offset + n_points]
        self.dataframe = df.iloc[offset : offset + n_points].reset_index(drop=True)
        return self.dataframe

    def get_close_prices(self):
        return self.dataframe["Close"].values

    def add_quantized_labels(self, n_bits: int = 50):
        """
        Fit a Quantization on the loaded closes, assign labels,
        and save the quantizer for later use.
        """
        closes    = self.get_close_prices()
        quantizer = Quantization(n_bits=n_bits)
        labels    = quantizer.fit_transform(closes)
        self.dataframe["label"] = labels
        self.quantizer = quantizer
        return labels
