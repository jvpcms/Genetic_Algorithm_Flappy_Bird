from .population import Population
from visualization import TraningTracker

from typing import Callable, Optional
from structures.neural_network import NeuralNetwork


class GeneticAlgorithm:
    population: Population
    generations: int
    fitness_function: Callable[[NeuralNetwork], float]
    training_tracker: Optional[TraningTracker]

    def __init__(
      self, 
      population: Population, 
      generations: int, 
      fitness_function: Callable[[NeuralNetwork], float],
      training_tracker: Optional[TraningTracker] = None
    ) -> 'GeneticAlgorithm':
      """
      Initialize the genetic algorithm object.

      Args:
          population: The population of individuals.
          generations: The number of generations.
          fitness_function: The fitness function.
          training_tracker: The training tracker.

      Returns:
          GeneticAlgorithm: The initialized genetic algorithm object.
      """

      self.population = population
      self.generations = generations
      self.fitness_function = fitness_function
      self.training_tracker = training_tracker

    def run(self):
      """
      Run the genetic algorithm.
      Performs the genetic algorithm process for the given number of generations.

      Args:
          None

      Returns:
          None
      """

      for individual in self.population.individuals:
        individual.fitness = self.fitness_function(individual)

      for _ in range(self.generations):

        self.population.survivor_of_the_fittest()

        self.population.reproduce()

        for individual in self.population.individuals:
          individual.fitness = self.fitness_function(individual)

        if self.training_tracker is not None:
          self.training_tracker.track(
            best_performance=max(individual.fitness for individual in self.population.individuals),
            average_performance=sum(individual.fitness for individual in self.population.individuals) / len(self.population.individuals)
          )