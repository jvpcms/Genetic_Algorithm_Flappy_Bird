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
        ) -> 'Population':
        """
        Initialize the population object.

        Args:
            size: The size of the population.
            layer_sizes: The sizes of the layers in the network.
            mutation_rate: The rate of mutation.
            survivor_rate: The rate of survivors.

        Returns:
            Population: The initialized population object.
        """
        self.individuals = [NeuralNetwork(layer_sizes) for _ in range(size)]

        self.size = size
        self.layer_sizes = layer_sizes
        self.mutation_rate = mutation_rate
        self.survivor_rate = survivor_rate

    def survivor_of_the_fittest(self) -> None:
        """
        Select the survivors of the fittest according to the ratio of survivors.
        Alters the individuals list in place.

        Args:
            None

        Returns:
            None
        """
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        self.individuals = self.individuals[:int(self.size * self.survivor_rate)]

    def selection(self) -> NeuralNetwork:
        """
        Roulette wheel method selection.
        Select a parent from the population according to the fitness of the individuals.
        Uses a weighted random selection.

        Args:
            None

        Returns:
            NeuralNetwork: The selected parent.
        """

        sum_fitness = sum(individual.fitness for individual in self.individuals)
        probabilities = [individual.fitness / sum_fitness for individual in self.individuals]
        r = np.random.rand()
        for i in range(len(probabilities)):
            r -= probabilities[i]
            if r <= 0:
                return self.individuals[i]
        return self.individuals[-1]

    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
        """
        Single point method crossover.
        Perform a crossover operation between two parents.
        Creates a new child network by combining the genome of the two parents.

        Args:
            parent1: The first parent.
            parent2: The second parent.

        Returns:
            NeuralNetwork: The child network.
        """

        split = np.random.randint(len(parent1.genome))
        child_genome = np.concatenate([parent1.genome[:split], parent2.genome[split:]])
        return NeuralNetwork(parent1.layer_sizes, genome=child_genome)

    def mutate(self, individual: NeuralNetwork, sigma: Optional[float] = 0.1) -> NeuralNetwork:
        """
        Mutate the individual's genome.
        Adds a small random value to the genome according to the mutation rate.

        Args:
            individual: The individual to mutate.
            sigma: The standard deviation of the random value.

        Returns:
            NeuralNetwork: The mutated individual.
        """

        if np.random.rand() <= self.mutation_rate:
            noise = np.random.standard_normal(individual.genome.shape) * sigma
            new_genome = individual.genome + noise
            return NeuralNetwork(individual.layer_sizes, genome=new_genome)

        return individual

    def reproduce(self) -> None:
        """
        Reproduce the population.
        Creates a new generation of individuals by performing crossover and mutation.

        Args:
            None

        Returns:
            None
        """

        new_gen = []

        for _ in range(self.size):
            parent1 = self.selection()
            parent2 = self.selection()

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_gen.append(child)

        self.individuals = new_gen
