import pytest
from atrax import Dataset


@pytest.fixture
def sample_dataset():
    data = [
        {"store": "B", "sales": 100, "profit": 20},
        {"store": "A", "sales": 200, "profit": 40},
        {"store": "C", "sales": None, "profit": 10},
        {"store": "A", "sales": 200, "profit": 30},
    ]
    return Dataset(data)

def test_sort_single_column_ascending(sample_dataset):
    sorted_ds = sample_dataset.sort_values(by="store", ascending=True)
    result = [row["store"] for row in sorted_ds.data]
    assert result == ["A", "A", "B", "C"]

def test_sort_single_column_descending(sample_dataset):
    sorted_ds = sample_dataset.sort_values(by="store", ascending=False)
    result = [row["store"] for row in sorted_ds.data]
    assert result == ["C", "B", "A", "A"]


def test_sort_multiple_columns(sample_dataset):
    sorted_ds = sample_dataset.sort_values(by=["store", "profit"], ascending=[True, True])
    result = [(row["store"], row["profit"]) for row in sorted_ds.data]
    assert result == [("A", 30), ("A", 40), ("B", 20), ("C", 10)]


def test_sort_with_na_position_first(sample_dataset):
    sorted_ds = sample_dataset.sort_values(by="sales", na_position="first")
    result = [row["sales"] for row in sorted_ds.data]
    assert result == [None, 100, 200, 200]


# def test_sort_with_na_position_last(sample_dataset):
#     sorted_ds = sample_dataset.sort_values(by="sales", na_position="last")
#     result = [row["sales"] for row in sorted_ds.data]
#     assert result == [100, 200, 200, None] 

def test_inplace_sort(sample_dataset):
    sample_dataset.sort_values(by="store", ascending=True, inplace=True)
    result = [row["store"] for row in sample_dataset.data]
    assert result == ["A", "A", "B", "C"]


def test_invalid_column_raises_keyerror(sample_dataset):
    with pytest.raises(KeyError):
        sample_dataset.sort_values(by="nonexistent")


def test_mismatched_by_and_ascending_raises_valueerror(sample_dataset):
    with pytest.raises(ValueError):
        sample_dataset.sort_values(by=["store", "sales"], ascending=[True])


def test_sort_descending_numeric(sample_dataset):
    sorted_ds = sample_dataset.sort_values(by="profit", ascending=False)
    result = [row["profit"] for row in sorted_ds.data]
    assert result == [40, 30, 20, 10]       