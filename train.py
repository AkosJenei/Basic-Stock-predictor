import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_processing import DataProcessor
from quantization import Quantization, build_codebook
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
CSV_PATH = "historical_data/AUDCHF_15m_historical_data.csv"
N_DATAPOINTS = 500
N_TESTPOINTS = 300
WINDOW = 3
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 100
OFFSET = 71500
BIN_SIZE = 0.001
USE_PRICE_CHANGES = False
QUANTIZER_OUT = "quantizer.pkl"

# Load data
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, n_points=N_DATAPOINTS, offset=OFFSET)
closes = dp.get_close_prices()
price_changes = dp.get_price_changes()
series = price_changes if USE_PRICE_CHANGES else closes

# Fit quantizer
TRAIN_SIZE = (len(series) if USE_PRICE_CHANGES else N_DATAPOINTS) - N_TESTPOINTS
quantizer = Quantization(bin_size=BIN_SIZE)
quantizer.fit(series[:TRAIN_SIZE])
M = quantizer.get_bits()
labels = quantizer.transform(series)
with open(QUANTIZER_OUT, "wb") as f:
    pickle.dump(quantizer, f)

# Build codebook
S = build_codebook(M)
L = S.shape[0]

# Build training sequences
one_hot = np.eye(M, dtype=np.float32)[labels[:TRAIN_SIZE]]
X_all = []
Y_all = []
label_targets = []
for t in range(WINDOW, TRAIN_SIZE):
    X_all.append(one_hot[t-WINDOW:t].reshape(-1))
    Y_all.append(S[:, labels[t]])
    label_targets.append(labels[t])
X_all = np.stack(X_all)
Y_all = np.stack(Y_all)
label_targets = np.array(label_targets)

# Train/validation split
X_train, X_val, Y_train, Y_val, lbl_train, lbl_val = train_test_split(
    X_all, Y_all, label_targets, test_size=TEST_SIZE, shuffle=True, random_state=42
)
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

# Model
model = create_model(input_dim=WINDOW * M, output_dim=L)
early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early],
    verbose=1,
)

# Evaluation
preds = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
bit_preds = np.where(preds >= 0, 1.0, -1.0)
S_T = S.T
label_preds = []
for bp in bit_preds:
    dists = ((S_T - bp) ** 2).sum(axis=1)
    label_preds.append(int(np.argmin(dists)))
label_preds = np.array(label_preds)

acc = accuracy_score(lbl_val, label_preds)
print(f"\nValidation accuracy: {acc:.4f}\n")
conf_mat = confusion_matrix(lbl_val, label_preds, labels=np.arange(M))
print("Confusion Matrix:")
print(conf_mat, "\n")
print("Classification Report:")
print(classification_report(lbl_val, label_preds, labels=np.arange(M), digits=4))

plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, aspect="auto")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.tight_layout()
plt.show()

