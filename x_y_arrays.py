import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from quantization import *

class x_y_arrays:
    def __init__(self, df, target, n=3):
        self.df = df
        self.Y = target[n:]
        self.X = self.create_sequences(df, n)
        assert len(self.X) == len(self.Y)

        quantizer = Quantization()
        bits = quantizer.get_bits()

        self.Y = to_categorical(self.Y, bits)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, shuffle=False)

    def create_sequences(data, n):
        sequence = []
        for i in range(0, len(data) - n):
            sequence.append(data[i:i+n].values)
        return np.array(sequence)
    
    def get_train_test(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def get_X(self):
        return self.X
    
    def get_Y(self):
        return self.Y
    
