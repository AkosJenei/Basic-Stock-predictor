import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from quantization import *

class x_y_arrays:
    """
    Class for creating input (X) and output (Y) arrays for machine learning.
    Handles sequence generation, one-hot encoding, and train-test splitting.
    """

    def __init__(self, df, target, n=3, test_size=0.2, shuffle=False, random_state=42):
        """
        Initialize the x_y_arrays object.

        df: input features (one-hot encoded or continuous)
        target: target labels
        n: sequence window size
        test_size: proportion of data to hold out for testing
        shuffle: whether to shuffle the train-test split
        random_state: random seed for reproducibility
        """
        self.X = self.create_sequences(df, n)
        self.Y = np.asarray(target)[n:]       

        if len(self.X) != len(self.Y):
            raise ValueError(f"Mismatch: X has {len(self.X)} samples but Y has {len(self.Y)}")

        bits = Quantization().get_bits()

        self.Y = to_categorical(self.Y, num_classes=bits)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X,
            self.Y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

    def create_sequences(self, data, n):
        """
        Slice data into overlapping sequences of length n.
        Returns a 3D array: (num_samples, n, num_features).
        """
        arr = data.values if hasattr(data, "values") else np.asarray(data)
        seqs = []
        for i in range(len(arr) - n):
            seqs.append(arr[i : i + n])
        return np.stack(seqs, axis=0)

    def get_train_test(self):
        """
        Return train-test split arrays: X_train, X_test, Y_train, Y_test.
        """
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def get_X(self):
        """
        Return the full X array (before train-test split).
        """
        return self.X

    def get_Y(self):
        """
        Return the full Y array (before train-test split).
        """
        return self.Y
