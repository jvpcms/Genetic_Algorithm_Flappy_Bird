from enum import Enum

from .classification_datasets import ConcentricCirclesDataset, NonLinearClustersDataset, BlobsDataset
from .classification_datasets import ClassificationDataset, DatasetType


def get_dataset(
  dataset_type: DatasetType, 
  features_size: int, 
  classes_size: int, 
  sample_size: int, 
  *, 
  train_test_split: float = 0.2
) -> ClassificationDataset:
  """
  Get a dataset based on the dataset type.
  """
  match dataset_type:
    case DatasetType.CONCENTRIC_CIRCLES:
      dataset = ConcentricCirclesDataset(features_size, classes_size, sample_size, train_test_split=train_test_split)
    case DatasetType.NON_LINEAR_CLUSTERS:
      dataset = NonLinearClustersDataset(features_size, classes_size, sample_size, train_test_split=train_test_split)
    case DatasetType.BLOBS:
      dataset = BlobsDataset(features_size, classes_size, sample_size, train_test_split=train_test_split)

  dataset.generate_data()
  return dataset