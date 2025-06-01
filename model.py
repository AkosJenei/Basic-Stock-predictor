from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.regularizers import l2

# Patch TensorFlow data adapter to avoid distributed dataset check errors
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

# Create and return a compiled LSTM model
def create_model(input_timesteps: int, n_classes: int) -> Sequential:
    model = Sequential([
        LSTM(256, activation="relu", input_shape=(input_timesteps, n_classes), return_sequences=True),
        LSTM(128, activation="relu", return_sequences=False),
        Dense(64, activation="relu", kernel_regularizer=l2(0.000124)),
        Dense(64, activation="relu", kernel_regularizer=l2(0.000124)),
        Dropout(0.2),
        Dense(n_classes, activation="softmax")
    ])
    optimizer = Adam(learning_rate=0.000285)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
