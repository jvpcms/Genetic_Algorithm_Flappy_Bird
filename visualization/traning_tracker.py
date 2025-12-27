from typing import List, Optional

from matplotlib import use as matplotlib_use
matplotlib_use('Qt5Agg')  # Use Qt5 interactive backend

import matplotlib.pyplot as plt


class TraningTracker:
  best_performance_history: List[float]
  average_performance_history: Optional[List[float]]

  def __init__(self) -> 'TraningTracker':
    """
    Initialize the training tracker object.

    Args:
        None

    Returns:
        TraningTracker: The initialized training tracker object.
    """

    self.best_performance_history = []
    self.average_performance_history = None

  def track(self, best_performance: float, average_performance: Optional[float] = None):
    """
    Track the best and average performance of the population.

    Args:
        best_performance: The best performance of the population.
        average_performance: The average performance of the population. If None, the average performance will not be tracked.

    Returns:
        None
    """
    print(f"Best performance: {best_performance}, Average performance: {average_performance}")
    self.best_performance_history.append(best_performance)

    if average_performance is not None:
      if self.average_performance_history is None:
        self.average_performance_history = []

      self.average_performance_history.append(average_performance)

  def plot(self):
    """
    Plot the best and average performance of the population.

    Args:
        None

    Returns:
        None
    """
    plt.plot(self.best_performance_history, label='Best Performance')
    if self.average_performance_history is not None:
      plt.plot(self.average_performance_history, label='Average Performance')
    plt.legend()
    plt.show()