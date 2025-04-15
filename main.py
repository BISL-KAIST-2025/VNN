import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from vnn_model import create_vnn_model


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, result_path, patience, epochs):
    """Train and evaluate the VNN model."""
    # Configure callbacks
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = f"{result_path}/best_model-{current_time}-{{val_loss:.2f}}.h5"
    
    callbacks = [
        ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
    ]

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, verbose=1, callbacks=callbacks)

    # Evaluate the model
    y_probs = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_probs)
    y_binary = (y_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_binary)
    return accuracy, auc_score


def main(data_path, result_path, patience, epochs, random_state):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    """Main function to run the VNN model pipeline."""
    # Load data
    gp2mf_adj = np.load(f"{data_path}/gp2mf_adj.npy")
    mf2bp_adj = np.load(f"{data_path}/mf2bp_adj.npy")
    X = np.load(f"{data_path}/embedding_x.npy", allow_pickle=True)
    y = np.load(f"{data_path}/embedding_y.npy", allow_pickle=True).astype(int)

    # Set configuration for VNN model
    input_shape = X.shape[1]
    n_gp = gp2mf_adj.shape[0]
    n_mf = mf2bp_adj.shape[0]
    n_bp = mf2bp_adj.shape[1]

    # Create the VNN model
    vnn_model = create_vnn_model(input_shape, n_gp, n_mf, n_bp, gp2mf_adj, mf2bp_adj)
    vnn_model.summary()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

    # Train and evaluate the model
    accuracy, auc_score = train_and_evaluate_model(
        vnn_model, X_train, X_test, y_train, y_test, result_path, patience, epochs
    )

    print(f"Accuracy score: {accuracy}")
    print(f"AUC Score: {auc_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VNN model pipeline.")
    parser.add_argument("--data_path", default='./data', help="Path to the input files.")
    parser.add_argument("--result_path", default='./result', help="Path to save the trained models.")
    parser.add_argument("--patience", type=int, default=32, help="Patience for early stopping.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to run.")
    parser.add_argument("--random_state", type=int, default=623, help="Random state for reproducibility.")
    
    args = parser.parse_args()
    main(args.data_path, args.result_path, args.patience, args.epochs, args.random_state)