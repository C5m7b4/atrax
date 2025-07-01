from atrax import Dataset, Series
import pytest
from test_data import data

def test_loc_setitem_boolean_series_scalar():
    ds = Dataset([
        {'name': 'Apple', 'price': 1.0},
        {'name': 'Steak', 'price': 10.0},
        {'name': 'Carrot', 'price': 0.5}
    ])

    mask = Series([False, True, False])
    ds.loc[mask, 'category'] = 'expensive'

    assert ds.data[0].get('category') is None
    assert ds.data[1].get('category') == 'expensive'
    assert ds.data[2].get('category') is None

def test_loc_setitem_boolean_series_list():
    ds = Dataset([
        {'a': 1}, {'a': 2}, {'a': 3}
    ])    
    mask = Series([True, False, True])
    ds.loc[mask, 'flag'] = ['x', 'y'] # seems confusing but we have 2 true values so our list just needs two values (x, y)

    assert ds.data[0]['flag'] == 'x'
    assert 'flag' not in ds.data[1]
    assert ds.data[2]['flag'] == 'y'

def test_loc_setitem_boolean_series_series():
    ds = Dataset([
        {'a': 10}, {'a': 20}, {'a': 30}
    ])
    mask = Series([False, True, True])
    val_series = Series([100, 200])
    ds.loc[mask, 'bonus'] = val_series

    assert 'bonus' not in ds.data[0]
    assert ds.data[1]['bonus'] == 100
    assert ds.data[2]['bonus'] == 200    

def test_loc_setitem_callable():
    ds = Dataset([
        {'name': 'Egg', 'price': 2},
        {'name': 'Milk', 'price': 6},
    ])

    ds.loc[lambda row: row['price'] > 3, 'label'] = lambda row: f"{row['name']} is expensive"

    assert 'label' not in ds.data[0]
    assert ds.data[1]['label'] == 'Milk is expensive'    


def test_loc_setitem_invalid_series_length():
    ds = Dataset([
        {'a': 1}, {'a': 2}, {'a': 3}
    ])
    mask = Series([True, False, True])
    val_series = Series([1])  # too short

    with pytest.raises(ValueError):
        ds.loc[mask, 'z'] = val_series  


def test_loc_setitem_missing_column():
    ds = Dataset([{'a': 1}])
    mask = Series([True])

    with pytest.raises(ValueError):
        ds.loc[mask] = 99  # Missing column name    

def test_loc_setitem_list_length_mismatch():
    ds = Dataset([
        {'name': 'apple', 'price': 1.0},
        {'name': 'steak', 'price': 10.0},
        {'name': 'banana', 'price': 2.0}
    ])

    # Boolean mask selecting 2 rows
    mask = Series([False, True, True])

    # Only 1 value, but 2 rows selected
    bad_values = ['expensive']

    with pytest.raises(ValueError, match="Length of list does not match number of selected rows"):
        ds.loc[mask, 'category'] = bad_values            

def test_loc_setitem_from_list():
    ds = Dataset([
        {'a': 10}, {'a': 20}, {'a': 30}
    ])
    mask = [False, True, True]
    val_series = Series([100, 200])
    ds.loc[mask, 'bonus'] = val_series

    assert 'bonus' not in ds.data[0]
    assert ds.data[1]['bonus'] == 100
    assert ds.data[2]['bonus'] == 200 

    

            