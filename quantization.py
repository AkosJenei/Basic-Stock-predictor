import numpy as np

class Quantization:
    def __init__(self, n_bits=40):
        self.n_bits = n_bits

    def get_bits(self):
        return self.n_bits