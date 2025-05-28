import numpy as np

class Quantization:
    def __init__(self, n_bits=50):

        self.n_bits = n_bits
        self.bin_edges = None

    def get_bits(self):
        return self.n_bits

    def fit(self, data):
        data = np.asarray(data)
        data_min, data_max = data.min(), data.max()
        self.bin_edges = np.linspace(data_min, data_max, self.n_bits + 1)

    def transform(self, data):
        if self.bin_edges is None:
            raise ValueError("Quantization bins not computed. "
                             "Call fit() first.")

        data = np.asarray(data)
        inner_edges = self.bin_edges[1:-1]
        codes = np.digitize(data, inner_edges, right=True)
        codes = np.clip(codes, 0, self.n_bits - 1)
        return codes

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, codes):
        if self.bin_edges is None:
            raise ValueError("Quantization bins not computed. "
                             "Call fit() first.")

        codes = np.asarray(codes)

        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        codes = np.clip(codes, 0, self.n_bits - 1)
        return centers[codes]
