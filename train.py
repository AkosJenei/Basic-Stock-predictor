from model import *
from x_y_arrays import *
from data_processing import *


def main():
    # Load your DataFrame here
    df = ...  # Replace with actual DataFrame loading code
    target = ...  # Replace with actual target column

    # Create x_y_arrays instance
    data = x_y_arrays(df, target)

    # Get training and testing data
    x_train, x_test, y_train, y_test = data.get_train_test()
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=1, validation_split=0.2, callbacks=[early_stopping])
