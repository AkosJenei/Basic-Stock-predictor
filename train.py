# train.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import *

from data_processing import DataProcessor
from x_y_arrays import x_y_arrays
from model import create_model
from quantization import Quantization

def main():
    CSV_PATH   = "historical_data/XAUUSD_historical_data_1h.csv"
    N_BITS     = 40
    WINDOW     = 3
    TEST_SIZE  = 0.2
    BATCH_SIZE = 64
    EPOCHS     = 100

    dp = DataProcessor()
    dp.read_csv(CSV_PATH)
    labels = dp.add_quantized_labels(n_bits=N_BITS)  
    one_hot = np.eye(N_BITS, dtype=int)[labels]  

    xy = x_y_arrays(
        df=one_hot, 
        target=labels, 
        n=WINDOW, 
        test_size=TEST_SIZE, 
        shuffle=True,
        random_state=42
    )
    X_train, X_test, Y_train, Y_test = xy.get_train_test()


    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print()

    model = create_model(input_timesteps=WINDOW, n_classes=N_BITS)
    model.summary()

    early = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, 
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early],
        verbose=1
    )

if __name__ == "__main__":
    main()
