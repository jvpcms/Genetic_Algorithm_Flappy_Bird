import numpy as np
from structures.neural_network import NeuralNetwork

from typing import List, Tuple, Optional

class Population:
    individuals: List[NeuralNetwork]
    size: int
    layer_sizes: Tuple[int, ...]

    mutation_rate: float
    survivor_rate: float

    def __init__(
            self, 
            size: int, 
            layer_sizes: Tuple[int, ...], 
            mutation_rate: float = 0.01, 
            survivor_rate: float = 0.1
        ):

        self.individuals = [NeuralNetwork(layer_sizes) for _ in range(size)]

        self.size = size
        self.layer_sizes = layer_sizes
        self.mutation_rate = mutation_rate
        self.survivor_rate = survivor_rate

    def survivor_of_the_fittest(self) -> None:
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        self.individuals = self.individuals[:int(self.size * self.survivor_rate)]

    def selection(self) -> NeuralNetwork:

        sum_fitness = sum(individual.fitness for individual in self.individuals)
        probabilities = [individual.fitness / sum_fitness for individual in self.individuals]
        r = np.random.rand()
        for i in range(len(probabilities)):
            r -= probabilities[i]
            if r <= 0:
                return self.individuals[i]
        return self.individuals[-1]

    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
        # Pick a random split point in the genome
        split = np.random.randint(len(parent1.genome))
        child_genome = np.concatenate([parent1.genome[:split], parent2.genome[split:]])
        return NeuralNetwork(parent1.layer_sizes, genome=child_genome)

    def mutate(self, individual: NeuralNetwork, sigma: Optional[float] = 0.1) -> NeuralNetwork:

        if np.random.rand() <= self.mutation_rate:
            noise = np.random.standard_normal(individual.genome.shape) * sigma
            new_genome = individual.genome + noise
            return NeuralNetwork(individual.layer_sizes, genome=new_genome)

        return individual

    def reproduce(self) -> None:

        new_gen = []

        for _ in range(self.size):
            parent1 = self.selection()
            parent2 = self.selection()

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_gen.append(child)

        self.individuals = new_gen
