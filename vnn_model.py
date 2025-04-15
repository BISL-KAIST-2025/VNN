import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam


class SparseDense(Layer):
    def __init__(self, units, adjacency_matrix, activation='relu', **kwargs):
        super(SparseDense, self).__init__(**kwargs)
        self.units = units
        self.adjacency_matrix = adjacency_matrix
        self.kernel_regularizer = tf.keras.regularizers.l2(0.000123)
        self.activation = tf.keras.activations.get(activation)
        self.batch_norm = BatchNormalization()

    def build(self, input_shape):
        # Create a trainable weight matrix with the same sparsity pattern as the adjacency matrix
        self.kernel = self.add_weight(
            "kernel", 
            shape=[int(input_shape[-1]), self.units],
            initializer="glorot_uniform",
            trainable=True,
            regularizer=self.kernel_regularizer
        )
        
        assert self.adjacency_matrix.shape == self.kernel.shape, "Adjacency matrix must have the same shape as the kernel"
        
        # Create a non-trainable mask based on the adjacency matrix
        self.mask = tf.constant(self.adjacency_matrix, dtype=self.kernel.dtype)

    def call(self, inputs):
        # Apply the mask to zero out weights
        masked_kernel = tf.multiply(self.kernel, self.mask)
        output = tf.matmul(inputs, masked_kernel)
        # output = self.batch_norm(output)
        output = self.activation(output)
        return output
    
    def get_config(self):
        config = super(SparseDense, self).get_config()
        config.update({
            'units': self.units,
            'adjacency_matrix': self.adjacency_matrix,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        activation = config.pop('activation')
        activation = tf.keras.activations.deserialize(activation)
        adjacency_matrix = config.pop('adjacency_matrix')
        adjacency_matrix = tf.convert_to_tensor(adjacency_matrix)
        return cls(**config, adjacency_matrix=adjacency_matrix, activation=activation)
    

def create_vnn_model(input_shape, n_gp, n_mf, n_bp, gp2mf_adj, mf2bp_adj):
    """Create a VNN model with SparseDense layers."""
    input_layer = Input(shape=(input_shape,))
    gp_layer = Dense(n_gp, activation='tanh')(input_layer)

    # SparseDense layers for the VNN model
    gp_mf_layer = SparseDense(units=n_mf, adjacency_matrix=gp2mf_adj, activation='tanh')(gp_layer)
    mf_bp_layer = SparseDense(units=n_bp, adjacency_matrix=mf2bp_adj, activation='tanh')(gp_mf_layer)

    # Output layer with sigmoid activation for binary classification
    output_layer = Dense(units=1, activation='sigmoid')(mf_bp_layer)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.0001, beta_1=0.999, beta_2=0.9)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model