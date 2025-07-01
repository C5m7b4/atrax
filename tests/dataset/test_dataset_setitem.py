from atrax import Dataset, Series
import pytest
from test_data import data

def test_set_column_from_list():
    ds = Dataset([{'a': 1}, {'a': 2}])
    ds['b'] = [10, 20]
    assert ds.data == [{'a': 1, 'b': 10}, {'a': 2, 'b': 20}]


def test_set_column_from_scalar():
    ds = Dataset([{'a': 1}, {'a': 2}])
    ds['flag'] = True
    assert all(row['flag'] is True for row in ds.data)  

def test_set_column_from_series():
    ds = Dataset([{'a': 1}, {'a': 2}])
    s = Series([True, False])
    ds['active'] = s
    assert ds.data[0]['active'] is True
    assert ds.data[1]['active'] is False    

def test_set_column_from_series_with_invalid_length():
    ds = Dataset([{'a': 1}, {'a': 2}])
    s = Series([True, False, True]) # too long
    
    with pytest.raises(ValueError):  
        ds['active'] = s  

def test_set_column_invalid_length():
    ds = Dataset([{'a': 1}, {'a': 2}])
    with pytest.raises(ValueError):
        ds['x'] = [1]  # too short      

def test_setitem_with_callable():
    ds = Dataset([
        {"name": "Apple", "price": 1.0},
        {"name": "Steak", "price": 10.0}
    ])

    # Assign a new column "expensive" based on a lambda
    ds["expensive"] = lambda row: row["price"] > 5

    assert ds.data[0]["expensive"] is False
    assert ds.data[1]["expensive"] is True   

def test_setitem_with_invalid_type_raises():
    ds = Dataset([
        {"a": 1},
        {"a": 2}
    ])

    # Attempt to assign a dict to a column – this should fail
    with pytest.raises(TypeError) as excinfo:
        ds["bad_column"] = {"x": 1}

    assert "Cannot assign value of type" in str(excinfo.value)         