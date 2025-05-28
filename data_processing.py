import pandas as pd
import numpy as np
from quantization import Quantization

class DataProcessor:
    def __init__(self, dataframe=None):
        self.dataframe = dataframe

    def read_csv(self, file_path, N_datapoints):
        self.dataframe = pd.read_csv(file_path)
        self.dataframe = self.dataframe.filter(
            ["open", "high", "low", "close", "tick_volume"]
        ).rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume"
        })
        self.dataframe = self.dataframe[-N_datapoints:-2000]
        return self.dataframe

    def get_dataframe(self):
        return self.dataframe

    def get_close_prices(self):
        return self.dataframe["Close"].values

    def add_quantized_labels(self, n_bits=50):
        close_prices = self.get_close_prices()
        quantizer = Quantization(n_bits=n_bits)
        labels = quantizer.fit_transform(close_prices)
        self.dataframe["label"] = labels
        return labels

    def get_labels(self):
        if "label" not in self.dataframe.columns:
            raise ValueError("No 'label' column – call add_quantized_labels() first.")
        return self.dataframe["label"].values

    def create_lagged_df(self, n=3):
        if "label" not in self.dataframe.columns:
            raise ValueError("No 'label' column – call add_quantized_labels() first.")
        labels = self.get_labels()
        bits = Quantization().get_bits()

        one_hot = np.eye(bits, dtype=int)[labels]

        records = []
        for t in range(n, len(one_hot)):
            row = {}
            for lag in range(n, 0, -1):
                row[f"i-{lag}"] = one_hot[t - lag]
            row["i"] = one_hot[t]
            records.append(row)

        return pd.DataFrame.from_records(records)
