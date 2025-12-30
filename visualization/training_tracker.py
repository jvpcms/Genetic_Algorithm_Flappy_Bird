from typing import List, Tuple, Callable, Optional
import numpy as np

from matplotlib import use as matplotlib_use
matplotlib_use('Qt5Agg')  # Use Qt5 interactive backend

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure

from datasets.classification_datasets import ClassificationDataset
from custom_logging import Logger

class TrainingTracker:

  classification_dataset: ClassificationDataset
  best_classifier: Callable[[np.ndarray], np.ndarray]

  maximum_performance: List[float] = []
  minimum_performance: List[float] = []
  average_performance: List[float] = []
  survivor_similarity_to_best: List[List[float]] = []

  logger: Optional[Logger]

  def __init__(self, classification_dataset: ClassificationDataset, logger: Optional[Logger] = None) -> None:
    """
    Initialize the training tracker object.

    Args:
        classification_dataset: The classification dataset.
        logger: The logger.

    Returns:
        None
    """

    self.classification_dataset = classification_dataset
    self.logger = logger
    if self.logger is not None:
      self.logger.info(f"Training tracker initialized")

  def track(
    self, 
    iteration: int, 
    maximum_performance: float, 
    minimum_performance: float, 
    average_performance: float,
    survivor_similarity_to_best: List[float]
  ) -> None:
    """
    Track the best and average performance of the population.

    Args:
        iteration: The current iteration/generation number.
        maximum_performance: The maximum (best) performance of the population.
        minimum_performance: The minimum (worst) performance of the population.
        average_performance: The average performance of the population.
        survivor_similarity_to_best: The similarity to the best survivor of the population at the current iteration.

    Returns:
        None
    """
    if self.logger is not None:
      self.logger.debug(f"Generation {iteration}: Max={maximum_performance:.4f}, Min={minimum_performance:.4f}, Avg={average_performance:.4f}")
    
    self.maximum_performance.append(maximum_performance)
    self.minimum_performance.append(minimum_performance)
    self.average_performance.append(average_performance)
    self.survivor_similarity_to_best.append(survivor_similarity_to_best)

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
      Training data shown as circles, test data shown as triangles.
      
      Args:
          None

      Returns:
         matplotlib.figure.Figure: The figure object.
      """
      resolution = 200
      # Auto-determine ranges from both train and test data
      all_X = np.vstack([self.classification_dataset.x_train, self.classification_dataset.x_test])
      x_range = (all_X[:, 0].min() - 0.5, all_X[:, 0].max() + 0.5)
      y_range = (all_X[:, 1].min() - 0.5, all_X[:, 1].max() + 0.5)
      
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
      
      # Overlay training data points as circles
      train_scatter = plt.scatter(
        self.classification_dataset.x_train[:, 0], 
        self.classification_dataset.x_train[:, 1], 
        c=self.classification_dataset.y_train, 
        cmap=plt.cm.Spectral, 
        edgecolors='black', 
        linewidths=1.5, 
        s=100, 
        marker='o',  # circles
        zorder=10,
        label='Training Data'
      )
      
      # Overlay test data points as triangles
      test_scatter = plt.scatter(
        self.classification_dataset.x_test[:, 0], 
        self.classification_dataset.x_test[:, 1], 
        c=self.classification_dataset.y_test, 
        cmap=plt.cm.Spectral, 
        edgecolors='black', 
        linewidths=1.5, 
        s=100, 
        marker='^',  # triangles
        zorder=10,
        label='Test Data'
      )
      
      plt.xlabel('Feature 1')
      plt.ylabel('Feature 2')
      plt.title('Decision Boundary with Training (circles) and Test (triangles) Data')
      plt.colorbar(train_scatter, label='Class')
      plt.legend()
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
      if self.logger is not None:
        self.logger.warning("No performance data to plot")
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

  def survivor_similarity_to_best_figure(self) -> MatplotlibFigure:
    """
    Plot the survivor similarity to the best survivor figure as a stacked area chart.

    The x-axis is the generation number, and the y-axis is the percentage (0-100%).
    At each generation, the amount of survivors in each 0.1 similarity range is counted and converted to percentages.
    The similarity plot shows stacked areas where each area represents the proportion of survivors in that similarity range.
    Ranges are inclusive of the top bound (e.g., (-0.9, -0.8] means -0.9 < x <= -0.8) except the lowest range which is inclusive of both bounds.
    Eg.: 
    - [-1.0, -0.9]: 10% of survivors (both bounds inclusive)
    - (-0.9, -0.8]: 20% of survivors (top bound inclusive)
    - (-0.8, -0.7]: 30% of survivors
    - ...
    - (0.8, 0.9]: 10% of survivors

    Args:
        None

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    if len(self.survivor_similarity_to_best) == 0:
      if self.logger is not None:
        self.logger.warning("No similarity data to plot")
      return None
    
    # Create similarity bin edges from -1.0 to 1.0 in 0.1 increments
    bin_edges = np.arange(-1.0, 1.1, 0.1)  # -1.0, -0.9, ..., 0.9, 1.0
    n_bins = len(bin_edges) - 1
    n_generations = len(self.survivor_similarity_to_best)
    
    # Create bin labels with proper inclusivity
    bin_labels = []
    for i in range(n_bins):
      if i == 0:
        # First bin: both bounds inclusive [-1.0, -0.9]
        bin_labels.append(f'[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]')
      else:
        # Other bins: top bound inclusive (-0.9, -0.8]
        bin_labels.append(f'({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]')
    
    # Create a 2D array to store percentages: [generation][similarity_bin]
    similarity_percentages = np.zeros((n_generations, n_bins))
    
    # Count survivors in each similarity bin for each generation and convert to percentages
    for gen_idx, similarities in enumerate(self.survivor_similarity_to_best):
      if len(similarities) > 0:
        # Manually assign to bins with correct inclusivity
        counts = np.zeros(n_bins)
        for sim in similarities:
          # Find which bin this similarity belongs to
          for bin_idx in range(n_bins):
            if bin_idx == 0:
              # First bin: [-1.0, -0.9] (both inclusive)
              if bin_edges[0] <= sim <= bin_edges[1]:
                counts[bin_idx] += 1
                break
            else:
              # Other bins: (bin_edges[i], bin_edges[i+1]] (top inclusive)
              if bin_edges[bin_idx] < sim <= bin_edges[bin_idx + 1]:
                counts[bin_idx] += 1
                break
        
        # Convert to percentages
        total = np.sum(counts)
        if total > 0:
          similarity_percentages[gen_idx, :] = (counts / total) * 100
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create stacked area chart
    # We'll stack from bottom to top, so we need cumulative sums
    cumulative = np.zeros(n_generations)
    
    # Use a colormap to assign colors to each bin
    colors = plt.cm.viridis(np.linspace(0, 1, n_bins))
    
    # Plot each bin as a stacked area (always plot, even if empty, to show in legend)
    for bin_idx in range(n_bins):
      percentages = similarity_percentages[:, bin_idx]
      plt.fill_between(
        range(n_generations),
        cumulative,
        cumulative + percentages,
        label=bin_labels[bin_idx],
        color=colors[bin_idx],
        alpha=0.7
      )
      cumulative += percentages
    
    # Set labels and title
    plt.xlabel('Generation')
    plt.ylabel('Percentage of Survivors (%)')
    plt.title('Survivor Similarity Distribution Over Generations (Stacked Area Chart)')
    plt.ylim(0, 100)
    plt.xlim(0, n_generations - 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add legend - show all bins
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, title='Similarity Range')
    
    plt.tight_layout()
    return plt.gcf()

  def plot(self):
    """
    Plot the performance, similarity, and decision boundary figures.

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
    
    # Show similarity plot
    sim_fig = self.survivor_similarity_to_best_figure()
    if sim_fig is not None:
      plt.figure(sim_fig.number)
      plt.show(block=False)
      plt.pause(0.1)  # Small pause to ensure window appears
    
    # Show decision boundary plot (only if best_classifier is set)
    if hasattr(self, 'best_classifier') and self.best_classifier is not None:
      db_fig = self.decision_boundary_figure()
      plt.figure(db_fig.number)
      plt.show(block=True)  # Block on the last plot
    else:
      self.logger.warning("Warning: best_classifier not set. Skipping decision boundary plot.")
      if perf_fig is not None or sim_fig is not None:
        plt.show(block=True)  # Block on other plots if no decision boundary
        
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

    for point, classification in zip(self.classification_dataset.x_train, self.classification_dataset.y_train):
      output = self.best_classifier(point)
      predicted_class = np.argmax(output.flatten())
      if predicted_class == classification:
        train_hits += 1

    train_accuracy = train_hits / len(self.classification_dataset.y_train)

    for point, classification in zip(self.classification_dataset.x_test, self.classification_dataset.y_test):
      output = self.best_classifier(point)
      predicted_class = np.argmax(output.flatten())
      if predicted_class == classification:
        test_hits += 1

    test_accuracy = test_hits / len(self.classification_dataset.y_test)

    return train_accuracy, test_accuracy
