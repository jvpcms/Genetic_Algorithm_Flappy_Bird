import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import (
  make_gaussian_quantiles, 
  make_moons, 
  make_swiss_roll, 
  make_blobs,
)
# X, y = make_moons(n_samples=100, noise=0.1)


class ClassificationDataset(ABC):
  train_size: int
  test_size: int

  features_size: int
  classes_size: int
  X_train: np.ndarray
  y_train: np.ndarray
  X_test: np.ndarray
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
        
    self.train_size = int(sample_size * (1 - train_test_split))
    self.test_size = int(sample_size * train_test_split)

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

    self.X_train, self.y_train = make_gaussian_quantiles(
      n_samples=self.train_size, 
      n_features=self.features_size,
      n_classes=self.classes_size,
    )
    self.X_test, self.y_test = make_gaussian_quantiles(
      n_samples=self.test_size, 
      n_features=self.features_size,
      n_classes=self.classes_size,
    )


class NonLinearClustersDataset(ClassificationDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def generate_data(self) -> None:
    """
    Generate the data for the non linear clusters dataset.
    Only supports 2 classes and 2 or more features.
    """
    self.X_train, self.y_train = make_moons(
      n_samples=self.train_size, 
      noise=self.noise
    )
    self.X_test, self.y_test = make_moons(
      n_samples=self.test_size, 
      noise=self.noise
    )

    if self.features_size > 2:
      self.X_train = np.hstack([self.X_train, np.random.normal(0, 0.1, (self.train_size, self.features_size - 2))])
      self.X_test = np.hstack([self.X_test, np.random.normal(0, 0.1, (self.test_size, self.features_size - 2))])
    elif self.features_size == 1:
      raise ValueError("Features size must be greater than 1")

    if self.classes_size != 2:
      raise ValueError("Only supports 2 classes")


class BlobsDataset(ClassificationDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def generate_data(self) -> None:
    """
    Generate the data for clusters in n-dimensional space dataset.
    """

    self.X_train, self.y_train = make_blobs(
      n_samples=self.train_size, 
      n_features=self.features_size,
      centers=self.classes_size,
    )
    self.X_test, self.y_test = make_blobs(
      n_samples=self.test_size, 
      n_features=self.features_size,
      centers=self.classes_size,
    )