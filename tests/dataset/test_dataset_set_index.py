from atrax import Dataset, Series
import pytest
from test_data import data

@pytest.fixture
def sample_dataset():
    """ A small 3-row Dataset for testing"""
    data = [
        {'id': 1, 'cat': 'A', 'val': 10},
        {'id': 2, 'cat': 'B', 'val': 20},
        {'id': 3, 'cat': 'C', 'val': 30},
    ]
    return Dataset(data)

def test_set_index_by_column_inplace(sample_dataset):
    ds = sample_dataset

    # act
    ret = ds.set_index('id', inplace=True)

    # assert we should not have returned anything because inplace=True
    assert ret is None
    # check the indexes
    assert ds._index == [1, 2, 3]
    assert ds._index_name == 'id'

    # original column retained (drop=False default)
    #assert 'id' in ds.columns

def test_set_index_by_column_new_object(sample_dataset):
    ds = sample_dataset

    # Act
    new_ds = ds.set_index("id", inplace=False)

    # Assert – returns new DataSet
    assert new_ds is not ds
    assert new_ds._index == [1, 2, 3]
    assert new_ds._index_name == "id"

    # Original untouched
    assert getattr(ds, "_index", None) != [1, 2, 3]   

def test_set_index_drop_column(sample_dataset):
    new_ds = sample_dataset.set_index("id", inplace=False, drop=True)

    # id column should be removed from each row and columns list
    assert "id" not in new_ds.columns
    assert all("id" not in row for row in new_ds.data)    

def test_set_index_with_callable(sample_dataset):
    f = lambda row: row["cat"].lower()
    new_ds = sample_dataset.set_index(f, inplace=False)

    assert new_ds._index == ["a", "b", "c"]
    assert new_ds._index_name is None  # anonymous index


def test_set_index_with_explicit_sequence(sample_dataset):
    seq = ["x", "y", "z"]
    ds = sample_dataset

    ds.set_index(seq, inplace=True)

    assert ds._index == seq
    assert ds._index_name is None   

# ---------------------------------------------------------------------------
# Error/edge cases
# ---------------------------------------------------------------------------

def test_set_index_missing_column(sample_dataset):
    with pytest.raises(KeyError):
        sample_dataset.set_index("does_not_exist")


def test_set_index_sequence_length_mismatch(sample_dataset):
    too_short = [1, 2]  # len != 3 rows
    with pytest.raises(ValueError):
        sample_dataset.set_index(too_short)


def test_set_index_invalid_type(sample_dataset):
    with pytest.raises(TypeError):
        sample_dataset.set_index(123)  # not str/callable/sequence      