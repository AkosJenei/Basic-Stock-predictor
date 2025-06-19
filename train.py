import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data_processing import DataProcessor
from quantization import Quantization
from x_y_arrays import x_y_arrays
from model import create_model, EarlyStopping

"""
Configuration:
- CSV_PATH: Path to historical data
- N_DATAPOINTS: Number of rows to load
- N_TESTPOINTS: Holdout size for backtesting
- WINDOW: Model input sequence length
- TEST_SIZE: Validation split ratio
- BATCH_SIZE: Training batch size
- EPOCHS: Training epochs
- OFFSET: Data window starting row
- N_BITS: Quantization levels
- QUANTIZER_OUT: Saved quantizer file
"""

CSV_PATH      = "historical_data/XAUUSD_4h_historical_data.csv"
N_DATAPOINTS  = 30000
N_TESTPOINTS  = 2000
WINDOW        = 3
TEST_SIZE     = 0.2
BATCH_SIZE    = 64
EPOCHS        = 100
OFFSET        = 22000
N_BITS        = 25

QUANTIZER_OUT = "quantizer.pkl"

"""
Load a segment of historical OHLCV data into a DataFrame.
"""
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, n_points=N_DATAPOINTS, offset=OFFSET)
closes = dp.get_close_prices()

"""
Fit quantizer on training data and transform the full dataset.
"""
TRAIN_SIZE = N_DATAPOINTS - N_TESTPOINTS
quantizer = Quantization(n_bits=N_BITS, bin_size=0.5)
quantizer.fit(closes[:TRAIN_SIZE])
num_classes = quantizer.get_bits()

labels = quantizer.transform(closes)

with open(QUANTIZER_OUT, "wb") as f:
    pickle.dump(quantizer, f)

"""
Prepare one-hot encoded sequences and labels for model training.
"""
one_hot = np.eye(num_classes, dtype=int)[labels[:TRAIN_SIZE]]
labels_train  = labels[:TRAIN_SIZE]

xy = x_y_arrays(
    df=one_hot,
    target=labels_train,
    n=WINDOW,
    test_size=TEST_SIZE,
    shuffle=True,
    random_state=42,
    num_classes=num_classes
)
X_train, X_val, Y_train, Y_val = xy.get_train_test()
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

"""
Define LSTM model, configure early stopping, and train.
"""
model = create_model(input_timesteps=WINDOW, n_classes=num_classes)
early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early],
    verbose=1
)

"""
Evaluate model accuracy, confusion matrix, and classification report.
"""
y_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
y_pred  = np.argmax(y_probs, axis=1)
y_true  = np.argmax(Y_val, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"\nValidation accuracy: {acc:.4f}\n")

all_labels = np.arange(num_classes)
conf_mat = confusion_matrix(y_true, y_pred, labels=all_labels)
print("Confusion Matrix:")
print(conf_mat, "\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, labels=all_labels, digits=4))

"""
Visualize the confusion matrix as a heatmap.
"""
plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, aspect='auto')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.colorbar()
plt.tight_layout()
plt.show()
