import pandas as pd
from quantization import Quantization

class DataProcessor:
    def __init__(self):
        self.dataframe = None  # stores the loaded and preprocessed DataFrame
        self.quantizer = None  # stores the Quantization object after fitting

    def read_csv(self, file_path: str, n_points: int, offset: int = 0):
        """
        Load exactly `n_points` rows starting from `offset` in a CSV file.
        Keeps only ['open', 'high', 'low', 'close', 'tick_volume'] columns 
        and renames them to ['Open', 'High', 'Low', 'Close', 'Volume'].
        Stores result in self.dataframe.
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
        self.dataframe = df.iloc[offset : offset + n_points].reset_index(drop=True)
        return self.dataframe

    def get_close_prices(self):
        """
        Return the 'Close' column as a numpy array.
        """
        return self.dataframe["Close"].values

    def add_quantized_labels(self, n_bits: int = 50):
        """
        Quantize the 'Close' column into `n_bits` categories.
        Add the resulting labels as a new 'label' column in self.dataframe.
        Store the fitted quantizer for future use.
        """
        closes    = self.get_close_prices()
        quantizer = Quantization(n_bits=n_bits)
        labels    = quantizer.fit_transform(closes)
        self.dataframe["label"] = labels
        self.quantizer = quantizer
        return labels
