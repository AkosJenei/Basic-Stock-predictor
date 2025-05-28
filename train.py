# train.py

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import matplotlib.pyplot as plt

from data_processing import DataProcessor
from x_y_arrays import x_y_arrays
from model import *

def build_train_model(X_train, Y_train, X_test, Y_test, N_BITS, WINDOW, BATCH_SIZE, EPOCHS):
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
    return model

CSV_PATH   = "historical_data/XAUUSD_15m_historical_data.csv"
N_datapoints = 5000
N_BITS     = 100
WINDOW     = 2
TEST_SIZE  = 0.2
BATCH_SIZE = 64
EPOCHS     = 100

# 1) Load & quantize
dp = DataProcessor()
dp.read_csv(CSV_PATH, N_datapoints)
labels  = dp.add_quantized_labels(n_bits=N_BITS)  
one_hot = np.eye(N_BITS, dtype=int)[labels]  # shape (T, 40)


xy = x_y_arrays(
        df=one_hot,
        target=labels,
        n=WINDOW,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=42
    )
X_train, X_test, Y_train, Y_test = xy.get_train_test()

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}\n")

model = build_train_model(X_train, Y_train, X_test, Y_test, N_BITS, WINDOW, BATCH_SIZE, EPOCHS)

y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.argmax(Y_test,      axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"\nTest accuracy: {acc:.4f}\n")

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm, "\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4))


plt.figure(figsize=(8, 6))
plt.imshow(cm, aspect='auto')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.colorbar()
plt.tight_layout()
plt.show()

