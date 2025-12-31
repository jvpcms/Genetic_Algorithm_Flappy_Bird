from .population import Population
from visualization import TrainingTracker

from typing import Callable, Optional
from structures.neural_network import NeuralNetwork
from custom_logging import Logger

class GeneticAlgorithm:
    population: Population
    generations: int
    fitness_function: Callable[[NeuralNetwork], float]
    training_tracker: Optional[TrainingTracker]

    logger: Optional[Logger]

    def __init__(
      self, 
      population: Population, 
      generations: int, 
      fitness_function: Callable[[NeuralNetwork], float],
      *,
      logger: Optional[Logger] = None,
      training_tracker: Optional[TrainingTracker] = None
      ) -> None:
      """
      Initialize the genetic algorithm object.

      Args:
          population: The population of individuals.
          generations: The number of generations.
          fitness_function: The fitness function.
          training_tracker: The training tracker. If None, the training tracker will not be used.

      Returns:
          None
      """

      self.population = population
      self.generations = generations
      self.fitness_function = fitness_function
      self.training_tracker = training_tracker
      self.logger = logger

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

      for i in range(self.generations):

        self.population.survivor_of_the_fittest()

        survivor_similarity_to_best = [
          survivor.similarity(self.population.survivors[0]) for survivor in self.population.survivors
        ]

        self.population.reproduce()

        for individual in self.population.individuals:
          individual.fitness = self.fitness_function(individual)

        if self.training_tracker is not None:
          self.training_tracker.track(
            iteration=i,
            maximum_performance=max(individual.fitness for individual in self.population.individuals),
            minimum_performance=min(individual.fitness for individual in self.population.individuals),
            average_performance=sum(individual.fitness for individual in self.population.individuals) / len(self.population.individuals),
            survivor_similarity_to_best=survivor_similarity_to_best,
          )

      if self.training_tracker is not None:
        self.training_tracker.set_best_classifier(self.population.individuals[0].predict)

        train_accuracy, test_accuracy = self.training_tracker.train_test_accuracy()
        if self.logger is not None:
          self.logger.info(f"Train accuracy: {train_accuracy:.4f}") 
          self.logger.info(f"Test accuracy: {test_accuracy:.4f}")

        self.training_tracker.plot()
