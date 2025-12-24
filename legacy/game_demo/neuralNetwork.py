import json
from generalFunctions import *


class NeuralNetwork:

    def __init__(self, layer_sizes, weights=None, biases=None):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]

        self.weights = weights
        self.biases = biases

        if weights is None:
            self.weights = [np.random.standard_normal(s) / np.sqrt(s[1]) for s in weight_shapes]
        if biases is None:
            self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.matmul(w, a) + b)

        return a

    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))


def generate_population(population_size, layer_sizes):
    population = [NeuralNetwork(layer_sizes) for _ in range(population_size)]
    return population


def new_generation(networks, performance, mutation_rate, n_children, layer_sizes):
    # selection: based in performance, better entities are selected more often;
    # cross: a portion of parametres comes from each parent;
    # mutation: randon standard distribution values added (or subtracted) from parametres.

    # normalize performance data
    for i in range(len(performance)):
        performance[i] = performance[i] ** 3
    total = sum(performance)
    for i in range(len(performance)):
        performance[i] /= total

    new_gen = []
    for i in range(n_children * len(networks)):
        p1 = weighted_choice(performance)
        p2 = weighted_choice(performance)
        parent1 = networks[p1]
        parent2 = networks[p2]

        # avarege_performance = (performance[p1] + performance[p2]) / 2

        child = crossover(parent1, parent2, layer_sizes)
        child = mutate(child, mutation_rate, layer_sizes)
        new_gen.append(child)

    return new_gen


def crossover(parent1, parent2, layer_sizes):
    child = parent2
    bias_split = int(np.random.rand() * sum(layer_sizes[:-1]))
    n_weights = 0

    for i in range(len(layer_sizes) - 1):
        n_weights += layer_sizes[i] * layer_sizes[i + 1]

    weight_split = int(np.random.rand() * n_weights)

    for bias in range(len(parent1.biases)):
        for b in range(len(parent1.biases[bias])):
            if bias_split > 0:
                child.biases[bias][b] = parent1.biases[bias][b]
            bias_split -= 1

    for weight in range(len(parent1.weights)):
        for line in range(len(parent1.weights[weight])):
            for col in range(len(parent1.weights[weight][line])):
                if weight_split > 0:
                    child.weights[weight][line][col] = parent1.weights[weight][line][col]
                weight_split -= 1

    return child


def mutate(network, mutation_rate, layer_sizes):

    if np.random.rand() <= mutation_rate:
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        weights = [np.random.standard_normal(s) / 100 for s in weight_shapes]
        biases = [np.random.standard_normal((s, 1)) / 100 for s in layer_sizes[1:]]

        weights = np.add(weights, network.weights)
        biases = np.add(biases, network.biases)

        return NeuralNetwork(layer_sizes, weights, biases)

    return network


def save_population(networks, size, layer_sizes, file_path):
    networks_data = {"size": size, "layer_sizes": layer_sizes, 'parametres': []}

    for n in range(len(networks)):
        networks_data['parametres'].append({})

        b = []
        w = []

        for s in range(1, len(layer_sizes)):
            b.append([])
            w.append([])

            for i in networks[n].biases[s - 1]:
                b[s - 1].append(i.tolist()[0])

            for i in range(layer_sizes[s]):
                w[s - 1].append([])
                for j in range(layer_sizes[s - 1]):
                    w[s - 1][i].append(networks[n].weights[s - 1][i][j])

        networks_data['parametres'][n]['weights'] = w
        networks_data['parametres'][n]['biases'] = b

    with open(file_path, 'w') as outfile:
        json.dump(networks_data, outfile, indent=4)


def load_population(file_path):
    with open(file_path, 'r') as infile:
        network_data = json.load(infile)

    networks = []
    size = network_data['size']
    layer_sizes = network_data['layer_sizes']

    for n in range(size):
        networks.append(NeuralNetwork(layer_sizes,
                                      network_data['parametres'][n]['weights'],
                                      network_data['parametres'][n]['biases']))

    return networks
