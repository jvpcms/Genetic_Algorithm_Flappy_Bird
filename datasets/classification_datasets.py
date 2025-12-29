import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from sklearn.datasets import (
  make_gaussian_quantiles, 
  make_moons, 
  make_swiss_roll, 
  make_blobs,
)

class DatasetType(Enum):
  CONCENTRIC_CIRCLES = "concentric_circles"
  NON_LINEAR_CLUSTERS = "non_linear_clusters"
  BLOBS = "blobs"


class ClassificationDataset(ABC):
  train_size: int
  test_size: int

  features_size: int
  classes_size: int
  x_train: np.ndarray
  y_train: np.ndarray
  x_test: np.ndarray
  y_test: np.ndarray

  noise: float = 0.1

  def __init__(
    self, 
    features_size: int, 
    classes_size: int, 
    sample_size: int,
    *,
    train_test_split: float = 0.2
  ):
    """
    Initialize the classification dataset.
    """
    self.features_size = features_size
    self.classes_size = classes_size
        
    self.test_size = int(sample_size * train_test_split)
    self.train_size = sample_size - self.test_size

  @abstractmethod
  def generate_data(self) -> None:
    """ 
    Generate the data for the dataset.
    """
    raise NotImplementedError("Subclasses must implement this method")


class ConcentricCirclesDataset(ClassificationDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def generate_data(self) -> None:
    """
    Generate the data for the concentric circles dataset.
    """

    X, Y = make_gaussian_quantiles(
      n_samples=self.train_size + self.test_size, 
      n_features=self.features_size,
      n_classes=self.classes_size,
    )

    self.x_train = X[:self.train_size]
    self.y_train = Y[:self.train_size]
    self.x_test = X[self.train_size:]
    self.y_test = Y[self.train_size:]


class NonLinearClustersDataset(ClassificationDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def generate_data(self) -> None:
    """
    Generate the data for the non linear clusters dataset.
    Only supports 2 classes and 2 or more features.
    """
    X, Y = make_moons(
      n_samples=self.train_size + self.test_size, 
      noise=self.noise
    )

    if self.features_size > 2:
      X = np.hstack([X, np.random.normal(0, 0.1, (self.train_size + self.test_size, self.features_size - 2))])
    elif self.features_size == 1:
      raise ValueError("Features size must be greater than 1")

    if self.classes_size != 2:
      raise ValueError("Only supports 2 classes")

    self.x_train = X[:self.train_size]
    self.y_train = Y[:self.train_size]
    self.x_test = X[self.train_size:]
    self.y_test = Y[self.train_size:]


class BlobsDataset(ClassificationDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def generate_data(self) -> None:
    """
    Generate the data for clusters in n-dimensional space dataset.
    """

    X, Y = make_blobs(
      n_samples=self.train_size + self.test_size, 
      n_features=self.features_size,
      centers=self.classes_size,
    )

    self.x_train = X[:self.train_size]
    self.y_train = Y[:self.train_size]
    self.x_test = X[self.train_size:]
    self.y_test = Y[self.train_size:]