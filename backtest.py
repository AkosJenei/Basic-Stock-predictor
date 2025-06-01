import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from train import model
from data_processing import DataProcessor

"""
Backtest configuration:
- CSV_PATH: Path to historical data
- N_DATAPOINTS: Number of rows to load for backtest
- N_TESTPOINTS: Size of test window
- WINDOW: Sequence window size for model input
- OFFSET: Row offset to start loading data from
- INITIAL_CAP: Starting capital for backtest
- LEVERAGE: Trade leverage
- RISK_PER_TRADE: % of capital risked per trade
- STOP_LOSS_PCT, TAKE_PROFIT_PCT: SL/TP levels (percent)
- QUANTIZER_PATH: Path to saved quantizer object
"""

CSV_PATH        = "historical_data/XAUUSD_4h_historical_data.csv"
N_DATAPOINTS    = 2000
N_TESTPOINTS    = 500
WINDOW          = 3
OFFSET          = 22000

INITIAL_CAP     = 5_000.0
LEVERAGE        = 100.0 
RISK_PER_TRADE  = 0.30
STOP_LOSS_PCT   = 1
TAKE_PROFIT_PCT = 1

QUANTIZER_PATH  = "quantizer.pkl"

"""
Load a specific window of historical OHLCV data into a DataFrame.
"""
dp = DataProcessor()
df = dp.read_csv(CSV_PATH, n_points=N_DATAPOINTS, offset=OFFSET)
closes = df["Close"].values
highs   = df["High"].values
lows    = df["Low"].values

"""
Apply the saved quantizer to map close prices to discrete labels.
Encode labels into one-hot format for model input.
"""
with open(QUANTIZER_PATH, "rb") as f:
    quantizer = pickle.load(f)

labels  = quantizer.transform(closes)
one_hot = np.eye(quantizer.get_bits(), dtype=int)[labels]

TRAIN_SIZE = N_DATAPOINTS - N_TESTPOINTS
start_idx  = TRAIN_SIZE + WINDOW

"""
Simulate a real-time backtest over the test set.
Predict signals bar by bar, execute trades with SL/TP logic.
Update capital and track trade returns.
"""
capital       = INITIAL_CAP
equity_curve  = [capital]
trade_returns = []
signals_list  = []
prev_label    = None

for t in range(start_idx, N_DATAPOINTS - 1):
    X_in   = one_hot[t-WINDOW:t][np.newaxis, :, :]
    y_prob = model.predict(X_in, verbose=0)[0]
    lbl    = int(np.argmax(y_prob))

    if prev_label is None:
        signal = 0
    else:
        signal = +1 if lbl > prev_label else -1 if lbl < prev_label else 0

    signals_list.append(signal)
    prev_label = lbl

    if signal != 0:
        entry = closes[t]
        tp    = entry*(1+TAKE_PROFIT_PCT) if signal>0 else entry*(1-TAKE_PROFIT_PCT)
        sl    = entry*(1-STOP_LOSS_PCT)   if signal>0 else entry*(1+STOP_LOSS_PCT)

        hi, lo, nxt = highs[t+1], lows[t+1], closes[t+1]
        if   signal>0 and hi  >= tp: exit_price = tp
        elif signal<0 and lo  <= tp: exit_price = tp
        elif signal>0 and lo  <= sl: exit_price = sl
        elif signal<0 and hi  >= sl: exit_price = sl
        else:                          exit_price = nxt

        margin   = capital * RISK_PER_TRADE
        notional = margin * LEVERAGE
        units    = notional / entry
        pnl      = units * ((exit_price-entry) if signal>0 else (entry-exit_price))

        capital += pnl
        trade_returns.append(pnl / margin)

    equity_curve.append(capital)

equity_curve  = np.array(equity_curve)
trade_returns = np.array(trade_returns)

"""
Calculate directional accuracy metrics:
- Confusion matrix of predicted vs actual signals
- Classification report
- Cumulative return and max drawdown
"""
test_labels     = labels[TRAIN_SIZE:]
true_dirs_full  = np.sign(np.diff(test_labels))
true_dirs       = true_dirs_full[WINDOW:]
pred_dirs       = np.array(signals_list)

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

"""
Visualize results:
- Equity curve over time
- Drawdown curve
- Histogram of individual trade returns
"""
plt.figure(figsize=(10,4))
plt.plot(equity_curve, label="Equity Curve for XAUUSD")
plt.title("Equity Curve for XAUUSD"); plt.grid(); plt.legend()

plt.figure(figsize=(10,4))
plt.plot(equity_curve/np.maximum.accumulate(equity_curve) - 1,
         color="red", label="Drawdown for XAUUSD")
plt.title("Drawdown Curve for XAUUSD"); plt.grid(); plt.legend()

plt.figure(figsize=(8,4))
plt.hist(trade_returns, bins=50)
plt.title("Distribution of Trade Returns for XAUUSD"); plt.grid()

plt.show()
