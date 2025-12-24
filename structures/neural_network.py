import numpy as np

from typing import List, Tuple, Optional


class NeuralNetwork:

    genome: np.ndarray
    layer_sizes: Tuple[int, ...]
    layers: List[Tuple[np.ndarray, np.ndarray]]
    fitness: float = 0

    def __init__(self, layer_sizes: Tuple[int, ...], genome: Optional[np.ndarray] = None):

        self.layer_sizes = layer_sizes

        total_genes = 0
        gene_shapes = []

        for i in range(len(layer_sizes) - 1):
            w_shape = (layer_sizes[i+1], layer_sizes[i])
            b_shape = (layer_sizes[i+1], 1)
            gene_shapes.append((w_shape, b_shape))
            total_genes += (layer_sizes[i+1] * layer_sizes[i]) + layer_sizes[i+1]

        if genome is None:
            self.genome = np.random.standard_normal(total_genes) / np.sqrt(total_genes)
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
        for w, b in self.layers:
            a = np.dot(w, a) + b
            a = self.activation(a)

        return a
    
    @staticmethod
    def activation(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def __repr__(self) -> str:
        result = "Neural Network:\n"
        for i, layer in enumerate(self.layers):
            result += f"Layer {i + 1}: {layer[0].shape} -> {layer[1].shape}\n"
            result += f"Weights: \n{layer[0]}\n"
            result += f"Biases: \n{layer[1]}\n"
            result += "\n"
        return result
    