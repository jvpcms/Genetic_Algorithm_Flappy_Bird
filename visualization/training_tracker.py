from typing import List, Tuple, Callable
import numpy as np

from matplotlib import use as matplotlib_use
matplotlib_use('Qt5Agg')  # Use Qt5 interactive backend

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure

from datasets.classification_datasets import ClassificationDataset

class TrainingTracker:

  classification_dataset: ClassificationDataset
  best_classifier: Callable[[np.ndarray], np.ndarray]

  maximum_performance: List[float] = []
  minimum_performance: List[float] = []
  average_performance: List[float] = []
  verbose: bool

  def __init__(self, classification_dataset: ClassificationDataset, verbose: bool = False) -> 'TrainingTracker':
    """
    Initialize the training tracker object.

    Args:
        classification_dataset: The classification dataset.
        verbose: Whether to print the tracked metrics to the console.

    Returns:
        TrainingTracker: The initialized training tracker object.
    """

    self.classification_dataset = classification_dataset
    self.verbose = verbose

    if self.verbose:
      print(f"Training tracker initialized with verbose mode: {self.verbose}")

  def track(
    self, 
    iteration: int, 
    maximum_performance: float, 
    minimum_performance: float, 
    average_performance: float
  ) -> None:
    """
    Track the best and average performance of the population.

    Args:
        iteration: The current iteration/generation number.
        maximum_performance: The maximum (best) performance of the population.
        minimum_performance: The minimum (worst) performance of the population.
        average_performance: The average performance of the population.

    Returns:
        None
    """
    if self.verbose:
      print(f"Generation {iteration}: Max={maximum_performance:.4f}, Min={minimum_performance:.4f}, Avg={average_performance:.4f}")
    
    self.maximum_performance.append(maximum_performance)
    self.minimum_performance.append(minimum_performance)
    self.average_performance.append(average_performance)
  
  def set_best_classifier(self, best_classifier: Callable[[np.ndarray], np.ndarray]):
    """
    Set the best classifier.

    Args:
        best_classifier: The best classifier.
    """
    self.best_classifier = best_classifier

  def decision_boundary_figure(self) -> MatplotlibFigure:
      """
      Plot the decision boundary of the best classifier.
      
      Args:
          None

      Returns:
         matplotlib.figure.Figure: The figure object.
      """
      resolution = 200
      # Auto-determine ranges from data if not provided
      x_range = (self.classification_dataset.X_train[:, 0].min() - 0.5, self.classification_dataset.X_train[:, 0].max() + 0.5)
      y_range = (self.classification_dataset.X_train[:, 1].min() - 0.5, self.classification_dataset.X_train[:, 1].max() + 0.5)
      
      # Create meshgrid
      x_min, x_max = x_range
      y_min, y_max = y_range
      
      xx = np.linspace(x_min, x_max, resolution)
      yy = np.linspace(y_min, y_max, resolution)
      X_mesh, Y_mesh = np.meshgrid(xx, yy)
      
      # Flatten the meshgrid for prediction
      grid_points = np.c_[X_mesh.ravel(), Y_mesh.ravel()]
      
      # Predict class for each point (quiet mode, no debug prints)
      predictions = []
      for point in grid_points:
          output = self.best_classifier(point)
          predicted_class = np.argmax(output.flatten())
          predictions.append(predicted_class)
      
      # Reshape predictions to match meshgrid
      Z = np.array(predictions).reshape(X_mesh.shape)
      
      # Create the plot
      plt.figure(figsize=(10, 8))
      
      # Show decision boundary
      plt.contourf(X_mesh, Y_mesh, Z, alpha=0.4, cmap=plt.cm.Spectral)
      plt.contour(X_mesh, Y_mesh, Z, colors='black', linewidths=0.5, alpha=0.2)
      
      # Overlay actual data points
      scatter = plt.scatter(
        self.classification_dataset.X_train[:, 0], 
        self.classification_dataset.X_train[:, 1], 
        c=self.classification_dataset.y_train, 
        cmap=plt.cm.Spectral, 
        edgecolors='black', 
        linewidths=1.5, 
        s=100, 
        zorder=10
      )
      
      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.title('Decision Boundary with Training Data Points')
      plt.colorbar(scatter, label='Class')
      plt.tight_layout()
      return plt.gcf()

  def performance_figure(self) -> MatplotlibFigure:
    """
    Plot the performance metrics.
    Average performance is plotted as a line, while maximum and minimum performance are plotted as shaded areas.

    Args:
        None

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    if len(self.average_performance) == 0:
      print("No performance data to plot")
      return None
    
    plt.figure(figsize=(10, 6))
    # Convert to numpy arrays for matplotlib
    iterations = np.arange(len(self.average_performance))
    avg_perf = np.array(self.average_performance)
    min_perf = np.array(self.minimum_performance)
    max_perf = np.array(self.maximum_performance)
    
    plt.plot(iterations, avg_perf, 'r-', label='Average Performance', linewidth=2)
    plt.fill_between(iterations, min_perf, max_perf, alpha=0.2, color='blue', label='Min-Max Range')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

  def plot(self):
    """
    Plot the performance and decision boundary figures.

    Args:
        None

    Returns:
        None
    """
    # Show performance plot
    perf_fig = self.performance_figure()
    if perf_fig is not None:
      plt.figure(perf_fig.number)
      plt.show(block=False)
      plt.pause(0.1)  # Small pause to ensure window appears
    
    # Show decision boundary plot (only if best_classifier is set)
    if hasattr(self, 'best_classifier') and self.best_classifier is not None:
      db_fig = self.decision_boundary_figure()
      plt.figure(db_fig.number)
      plt.show(block=True)  # Block on the last plot
    else:
      print("Warning: best_classifier not set. Skipping decision boundary plot.")
      if perf_fig is not None:
        plt.show(block=True)  # Block on performance plot if no decision boundary
        
  def train_test_accuracy(self) -> Tuple[float, float]:
    """
    Compute the train and test accuracy of the best classifier.

    Args:
        None

    Returns:
        Tuple[float, float]: The train and test accuracy.
    """

    train_hits = 0
    test_hits = 0

    for point, classification in zip(self.classification_dataset.X_train, self.classification_dataset.y_train):
      output = self.best_classifier(point)
      predicted_class = np.argmax(output.flatten())
      if predicted_class == classification:
        train_hits += 1

    train_accuracy = train_hits / len(self.classification_dataset.y_train)

    for point, classification in zip(self.classification_dataset.X_test, self.classification_dataset.y_test):
      output = self.best_classifier(point)
      predicted_class = np.argmax(output.flatten())
      if predicted_class == classification:
        test_hits += 1

    test_accuracy = test_hits / len(self.classification_dataset.y_test)

    return train_accuracy, test_accuracy