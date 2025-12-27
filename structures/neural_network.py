import numpy as np

from typing import List, Tuple, Optional


class NeuralNetwork:

    genome: np.ndarray
    layer_sizes: Tuple[int, ...]
    layers: List[Tuple[np.ndarray, np.ndarray]]
    fitness: float = 0

    def __init__(self, layer_sizes: Tuple[int, ...], genome: Optional[np.ndarray] = None) -> 'NeuralNetwork':
        """
        Initialize the neural network object.

        Args:
            layer_sizes: The sizes of the layers in the network.
            genome: The genome of the network. If None, the genome will be initialized randomly.

        Returns:
            NeuralNetwork: The initialized neural network object.
        """

        self.layer_sizes = layer_sizes

        total_genes = 0
        gene_shapes = []

        for i in range(len(layer_sizes) - 1):
            w_shape = (layer_sizes[i+1], layer_sizes[i])
            b_shape = (layer_sizes[i+1], 1)
            gene_shapes.append((w_shape, b_shape))
            total_genes += (layer_sizes[i+1] * layer_sizes[i]) + layer_sizes[i+1]

        if genome is None:
            # Use Xavier/Glorot initialization for better weight distribution
            self.genome = np.random.standard_normal(total_genes)
            # Scale weights by layer size (Xavier initialization)
            idx = 0
            for w_shape, b_shape in gene_shapes:
                w_size = np.prod(w_shape)
                b_size = np.prod(b_shape)
                # Xavier initialization: scale by sqrt of input size
                fan_in = w_shape[1]
                self.genome[idx:idx + w_size] /= np.sqrt(fan_in)
                idx += w_size + b_size
        else:
            self.genome = genome

        self.layers = []
        current_idx = 0
        for w_shape, b_shape in gene_shapes:
            w_size = np.prod(w_shape)
            b_size = np.prod(b_shape)
            
            w_view = self.genome[current_idx : current_idx + w_size].reshape(w_shape)
            current_idx += w_size
            b_view = self.genome[current_idx : current_idx + b_size].reshape(b_shape)
            current_idx += b_size
            
            self.layers.append((w_view, b_view))


    def predict(self, a: np.ndarray) -> np.ndarray:
        """
        Feed input through the network and return the output.

        Args:
            a: The input to the network.
              - Must be a column vector.
              - Must be of shape (n_features, 1).

        Returns:
            np.ndarray: The output of the network.
              - Must be a column vector.
              - Must be of shape (n_outputs, 1).
        """
        # Ensure input is a column vector
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        
        for w, b in self.layers:
            a = np.dot(w, a) + b
            a = self.activation(a)

        return a
    
    @staticmethod
    def activation(x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function to the input.

        Args:
            x: The input to the activation function.

        Returns:
            np.ndarray: The output of the activation function.
              - Must be a column vector.
              - Must be of shape (n_outputs, 1).
        """
        return 1 / (1 + np.exp(-x))

    def __repr__(self) -> str:
        """
        Return a string representation of the neural network.

        Returns:
            str: A string representation of the neural network.
        """
        result = "Neural Network:\n"
        for i, layer in enumerate(self.layers):
            result += f"Layer {i + 1}: {layer[0].shape} -> {layer[1].shape}\n"
            result += f"Weights: \n{layer[0]}\n"
            result += f"Biases: \n{layer[1]}\n"
            result += "\n"
        return result
    