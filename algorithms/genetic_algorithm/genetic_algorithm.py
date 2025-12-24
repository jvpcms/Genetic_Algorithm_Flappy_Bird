from .population import Population

from typing import Callable
from structures.neural_network import NeuralNetwork


class GeneticAlgorithm:
    population: Population
    generations: int
    fitness_function: Callable[[NeuralNetwork], float]

    def __init__(self, population: Population, generations: int, fitness_function: Callable[[NeuralNetwork], float]):
        self.population = population
        self.generations = generations
        self.fitness_function = fitness_function

    def run(self):
      for _ in range(self.generations):
        for individual in self.population.individuals:
          individual.fitness = self.fitness_function(individual)

        self.population.reproduce()