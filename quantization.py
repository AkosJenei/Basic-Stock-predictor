import numpy as np

class Quantization:
    def __init__(self, n_bits=25):
        """Initialize the quantizer with a specified number of bins (n_bits)."""
        self.n_bits = n_bits
        self.bin_edges = None

    def get_bits(self):
        """Return the number of quantization bins (useful for model configuration)."""
        return self.n_bits

    def fit(self, data):
        """
        Compute the bin edges for quantization based on input data.
        Divides the range [min, max] into equal-width intervals.
        """
        data = np.asarray(data)
        data_min, data_max = data.min(), data.max()
        self.bin_edges = np.linspace(data_min, data_max, self.n_bits + 1)

    def transform(self, data):
        """
        Quantize the input data into integer bin codes [0, n_bits-1].
        Each data point is mapped to a bin based on its value.
        """
        if self.bin_edges is None:
            raise ValueError("Quantization bins not computed. Call fit() first.")

        data = np.asarray(data)
        inner_edges = self.bin_edges[1:-1]
        codes = np.digitize(data, inner_edges, right=True)
        codes = np.clip(codes, 0, self.n_bits - 1)
        return codes

    def fit_transform(self, data):
        """
        Fit the quantizer on the data and return the quantized bin codes.
        Convenience method combining fit() and transform().
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, codes):
        """
        Convert quantized bin codes back to approximate continuous values (bin centers).
        Returns the center of the corresponding bin for each code.
        """
        if self.bin_edges is None:
            raise ValueError("Quantization bins not computed. Call fit() first.")

        codes = np.asarray(codes)
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        codes = np.clip(codes, 0, self.n_bits - 1)
        return centers[codes]
