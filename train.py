# train.py

import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

from data_processing import DataProcessor
from quantization import Quantization
from x_y_arrays import x_y_arrays
from model import create_model, EarlyStopping

# -----------------------------------------------------------------------------
# Configurable parameters
# -----------------------------------------------------------------------------
CSV_PATH      = "historical_data/XAUUSD_4h_historical_data.csv"
N_DATAPOINTS  = 2000    # total rows in this slice (train + holdout)
N_TESTPOINTS  = 500     # last 500 for backtesting later
WINDOW        = 3
TEST_SIZE     = 0.2     # validation split within training slice
BATCH_SIZE    = 64
EPOCHS        = 100
OFFSET        = 20000      # start row of our 5k-window in the full CSV
N_BITS        = 25

QUANTIZER_OUT = "quantizer.pkl"

# -----------------------------------------------------------------------------
# 1) Load the 5k-row slice [OFFSET : OFFSET+5000]
# -----------------------------------------------------------------------------
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, n_points=N_DATAPOINTS, offset=OFFSET)
closes = dp.get_close_prices()

# -----------------------------------------------------------------------------
# 2) Fit quantizer on the first (N_DATAPOINTS - N_TESTPOINTS) rows
# -----------------------------------------------------------------------------
TRAIN_SIZE = N_DATAPOINTS - N_TESTPOINTS
quantizer  = Quantization(n_bits=N_BITS)
quantizer.fit(closes[:TRAIN_SIZE])

# apply to entire slice
labels = quantizer.transform(closes)

# save quantizer for backtest
with open(QUANTIZER_OUT, "wb") as f:
    pickle.dump(quantizer, f)

# -----------------------------------------------------------------------------
# 3) Prepare training data (only on first TRAIN_SIZE samples)
# -----------------------------------------------------------------------------
one_hot       = np.eye(N_BITS, dtype=int)[labels[:TRAIN_SIZE]]
labels_train  = labels[:TRAIN_SIZE]

xy = x_y_arrays(
    df=one_hot,
    target=labels_train,
    n=WINDOW,
    test_size=TEST_SIZE,
    shuffle=True,
    random_state=42
)
X_train, X_val, Y_train, Y_val = xy.get_train_test()
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

# -----------------------------------------------------------------------------
# 4) Build & train model
# -----------------------------------------------------------------------------
model = create_model(input_timesteps=WINDOW, n_classes=N_BITS)
early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early],
    verbose=1
)

# -----------------------------------------------------------------------------
# 5) Evaluate on validation set
# -----------------------------------------------------------------------------
y_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
y_pred  = np.argmax(y_probs, axis=1)
y_true  = np.argmax(Y_val, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"\nValidation accuracy: {acc:.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred), "\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4))