import numpy as np
import math


class Quantization:
    def __init__(self, n_bits=25, bin_size=None):
        """Initialize the quantizer.

        Parameters
        ----------
        n_bits : int, optional
            Number of quantization bins to use when ``bin_size`` is not
            provided. Defaults to ``25`` to preserve existing behaviour.
        bin_size : float, optional
            Width of each bin. When supplied, ``n_bits`` will be ignored and
            the number of bins will be computed during ``fit``.
        """

        if n_bits is None and bin_size is None:
            raise ValueError("Either n_bits or bin_size must be specified")

        self.n_bits = n_bits
        self.bin_size = bin_size
        self.bin_edges = None

    def get_bits(self):
        """Return the number of quantization bins (useful for model configuration)."""
        return self.n_bits

    def fit(self, data):
        """Compute bin edges from ``data``.

        When ``bin_size`` is specified, bin edges are created at that fixed
        interval. Otherwise the range ``[min, max]`` is divided into
        ``n_bits`` equal-width bins.
        """

        data = np.asarray(data)
        data_min, data_max = data.min(), data.max()

        if self.bin_size is not None:
            # Use explicit bin width
            self.bin_edges = np.arange(data_min, data_max + self.bin_size, self.bin_size)
            # Update n_bits based on the derived number of bins
            self.n_bits = len(self.bin_edges) - 1
        else:
            self.bin_edges = np.linspace(data_min, data_max, self.n_bits + 1)

    def transform(self, data):
        """Quantize ``data`` into integer bin codes [0, ``n_bits``-1]."""
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
        """Map integer codes back to approximate continuous values."""
        if self.bin_edges is None:
            raise ValueError("Quantization bins not computed. Call fit() first.")

        codes = np.asarray(codes)
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        codes = np.clip(codes, 0, self.n_bits - 1)
        return centers[codes]


def build_codebook(n_bins: int) -> np.ndarray:
    """Return an L x M codebook splitting bins hierarchically."""
    L = int(math.ceil(math.log2(n_bins)))
    S = np.empty((L, n_bins), dtype=np.float32)
    for m in range(n_bins):
        for l in range(L):
            bit = (m >> (L - l - 1)) & 1
            S[l, m] = 1.0 if bit == 0 else -1.0
    return S
