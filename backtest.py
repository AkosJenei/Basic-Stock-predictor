import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from train import model
from data_processing import DataProcessor

# -----------------------------------------------------------------------------
# Configurable parameters
# -----------------------------------------------------------------------------
CSV_PATH        = "historical_data/XAUUSD_4h_historical_data.csv"
N_DATAPOINTS    = 2000    # total rows in this slice
N_TESTPOINTS    = 500     # last 500 rows for backtest
WINDOW          = 3
OFFSET          = 25000       # start row of the 5k-slice in the full CSV

INITIAL_CAP     = 5_000.0
LEVERAGE        = 100.0 
RISK_PER_TRADE  = 0.30
STOP_LOSS_PCT   = 0.002
TAKE_PROFIT_PCT = 0.002

QUANTIZER_PATH  = "quantizer.pkl"

# -----------------------------------------------------------------------------
# 1) Load the 5k-row slice [OFFSET : OFFSET+5000]
# -----------------------------------------------------------------------------
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, n_points=N_DATAPOINTS, offset=OFFSET)
closes = df["Close"].values
highs   = df["High"].values
lows    = df["Low"].values

# -----------------------------------------------------------------------------
# 2) Load pre-fitted quantizer & transform closes → labels → one-hot
# -----------------------------------------------------------------------------
with open(QUANTIZER_PATH, "rb") as f:
    quantizer = pickle.load(f)

labels  = quantizer.transform(closes)
one_hot = np.eye(quantizer.get_bits(), dtype=int)[labels]

# split sizes
TRAIN_SIZE = N_DATAPOINTS - N_TESTPOINTS
start_idx  = TRAIN_SIZE + WINDOW   # first index where we backtest

# -----------------------------------------------------------------------------
# 3) Pre-allocate pred_dirs to match true_dirs length
# -----------------------------------------------------------------------------
test_labels     = labels[TRAIN_SIZE:]                  # length = N_TESTPOINTS = 500
true_dirs_full  = np.sign(np.diff(test_labels))        # length = 499
true_dirs       = true_dirs_full[WINDOW:]              # length = 496
N_PREDICT       = len(true_dirs)                       # 496

pred_dirs       = np.zeros(N_PREDICT, dtype=int)
capital         = INITIAL_CAP
equity_curve    = [capital]
trade_returns   = []
prev_label      = None

t = start_idx
while t < (N_DATAPOINTS - 1):
    # 3.a) Form input for the model at time t
    X_in   = one_hot[t-WINDOW:t][np.newaxis, :, :]
    y_prob = model.predict(X_in, verbose=0)[0]
    lbl    = int(np.argmax(y_prob))

    # 3.b) Generate +1/−1/0 signal based on label change
    if prev_label is None:
        signal = 0
    else:
        signal = +1 if lbl > prev_label else -1 if lbl < prev_label else 0

    # 3.c) Store signal into pred_dirs at index i
    i = t - start_idx
    if 0 <= i < N_PREDICT:
        pred_dirs[i] = signal
    prev_label = lbl

    # 3.d) If no signal, just advance one bar
    if signal == 0:
        equity_curve.append(capital)
        t += 1
        continue

    # 3.e) If we have a nonzero signal, open a new trade at close[t]
    entry_price = closes[t]
    if signal > 0:
        tp_level = entry_price * (1 + TAKE_PROFIT_PCT)
        sl_level = entry_price * (1 - STOP_LOSS_PCT)
    else:  # short
        tp_level = entry_price * (1 - TAKE_PROFIT_PCT)
        sl_level = entry_price * (1 + STOP_LOSS_PCT)

    # We allocate risk and compute units once at entry time:
    margin   = capital * RISK_PER_TRADE
    notional = margin * LEVERAGE
    units    = notional / entry_price

    # 3.f) Now scan forward, bar by bar, until TP or SL is hit,
    #      or until we hit the end of our test period.
    exit_price = None
    exit_bar   = None
    u = t + 1
    while u < N_DATAPOINTS:
        h = highs[u]
        l = lows[u]
        c = closes[u]

        if signal > 0:
            # LONG: check TP first, then SL
            if h >= tp_level:
                exit_price = tp_level
                exit_bar   = u
                break
            elif l <= sl_level:
                exit_price = sl_level
                exit_bar   = u
                break
        else:
            # SHORT: check TP first (price has to go down), then SL
            if l <= tp_level:
                exit_price = tp_level
                exit_bar   = u
                break
            elif h >= sl_level:
                exit_price = sl_level
                exit_bar   = u
                break

        # If neither level was hit, move to next bar
        u += 1

    # 3.g) If we ran out of bars without hitting TP/SL, exit at final close
    if exit_price is None:
        exit_bar   = N_DATAPOINTS - 1
        exit_price = closes[exit_bar]

    # 3.h) Compute P&L and update capital
    if signal > 0:
        pnl = units * (exit_price - entry_price)
    else:
        pnl = units * (entry_price - exit_price)

    capital += pnl
    trade_returns.append(pnl / margin)

    # 3.i) Fill equity_curve for each bar held:
    #       - equity is flat up until exit_bar−1 (we already appended last equity on entry or previous)
    #       - append one entry at exit_bar to reflect new capital
    equity_curve.append(capital)

    # 3.j) Jump t to the bar after exit_bar
    t = exit_bar + 1
    # Note: prev_label remains as whatever label we last saw at time t

# Convert equity_curve and trade_returns to numpy arrays
equity_curve  = np.array(equity_curve)
trade_returns = np.array(trade_returns)

# -----------------------------------------------------------------------------
# 4) Metrics & Reporting
# -----------------------------------------------------------------------------
# (true_dirs and pred_dirs are already defined above)
if len(true_dirs) != len(pred_dirs):
    raise ValueError(f"Length mismatch: true_dirs={len(true_dirs)}, pred_dirs={len(pred_dirs)}")

cm = confusion_matrix(true_dirs, pred_dirs, labels=[-1,0,1])
print("=== Directional Accuracy ===")
print("Confusion Matrix (-1,0,+1):")
print(cm)
print("\nClassification Report:")
print(classification_report(true_dirs, pred_dirs, labels=[-1,0,1]))

cum_return = equity_curve[-1]/INITIAL_CAP - 1
max_dd     = (equity_curve/np.maximum.accumulate(equity_curve) - 1).min()
print(f"\n=== Performance Summary ===")
print(f"Cumulative Return: {cum_return:.2%}")
print(f"Max Drawdown:     {max_dd:.2%}")

# -----------------------------------------------------------------------------
# 5) Plots
# -----------------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(equity_curve, label="Equity Curve")
plt.title("Equity Curve"); plt.grid(); plt.legend()

plt.figure(figsize=(10,4))
plt.plot(equity_curve/np.maximum.accumulate(equity_curve) - 1,
         color="red", label="Drawdown")
plt.title("Drawdown Curve"); plt.grid(); plt.legend()

plt.figure(figsize=(8,4))
plt.hist(trade_returns, bins=50)
plt.title("Distribution of Trade Returns"); plt.grid()

plt.show()
