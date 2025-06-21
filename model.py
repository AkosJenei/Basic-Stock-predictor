from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizer_v2.adam import Adam
from tensorflow.keras.engine import data_adapter

# Patch TensorFlow data adapter to avoid distributed dataset check errors
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset


def create_model(input_dim: int, output_dim: int) -> Sequential:
    """Create and compile a simple feed-forward network."""
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        Dense(output_dim)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")
    return model

