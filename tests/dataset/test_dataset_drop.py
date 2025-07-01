import pytest

from atrax import Dataset

# Fixtures
@pytest.fixture
def sample_dataset():
    """A small 3-row DataSet for testing."""
    data = [
        {"id": 1, "cat": "A", "val": 10},
        {"id": 2, "cat": "B", "val": 20},
        {"id": 3, "cat": "C", "val": 30},
    ]
    return Dataset(data)


# ---------------------------------------------------------------------------
# drop tests 
# ---------------------------------------------------------------------------

def test_drop_rows_inplace(sample_dataset):
    ds = sample_dataset
    ds.set_index("id", inplace=True)

    ds.drop(index=[1], inplace=True)  # drop the second row (pos 1, id==2)

    # dataset shrunk & index updated
    assert len(ds.data) == 2
    assert ds._index == [1, 3]
    # ensure id 2 gone
    # assert all(row["id"] != 2 for row in ds.data)


def test_drop_rows_new_object(sample_dataset):
    ds = sample_dataset
    ds.set_index("id", inplace=True)

    new_ds = ds.drop(index=[0, 2], inplace=False)  # keep only middle row

    assert len(new_ds.data) == 1
    assert new_ds._index == [2]
    # original untouched
    assert len(ds.data) == 3 and ds._index == [1, 2, 3]


def test_drop_columns_inplace(sample_dataset):
    ds = sample_dataset
    ds.drop(columns=["cat"], inplace=True)

    assert "cat" not in ds.columns
    assert all("cat" not in row for row in ds.data)


def test_drop_columns_new_object(sample_dataset):
    ds = sample_dataset
    new_ds = ds.drop(columns="val", inplace=False)  # accept str

    assert "val" not in new_ds.columns
    # original intact
    assert "val" in ds.columns


def test_drop_rows_and_columns(sample_dataset):
    ds = sample_dataset
    ds.set_index("id", inplace=True)

    new_ds = ds.drop(columns=["cat"], index=[1], inplace=False)  # drop 2nd row & cat col

    assert len(new_ds.data) == 2
    assert "cat" not in new_ds.columns
    assert new_ds._index == [1, 3]


def test_drop_invalid_column_raises(sample_dataset):
    with pytest.raises(KeyError):
        sample_dataset.drop(columns=["does_not_exist"])
   
