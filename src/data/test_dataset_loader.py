import pytest
import dataset_loader as dl
from datasets import Dataset
import logging

# Unit tests using pytests
def test_load_hf_dataset_with_sampling():
    n_samples = 100
    dataset = dl.load_hf_dataset_with_sampling('glue,ax,test', n_samples=n_samples)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == n_samples

def test_load_hf_dataset_with_sampling_invalid_name():
    with pytest.raises(ValueError):
        dl.load_hf_dataset_with_sampling('nonexistent_dataset', n_samples=None)
