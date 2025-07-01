from atrax import Dataset
import pytest

ds = Dataset({
    'name': ['alice', 'bob', 'charlie'],
    'age': [25, 30, 35],
    'score': [88.5, 92.0, 95.0]
}, index=['a', 'b', 'c'])


# iloc tests
def test_iloc_with_single_row():
    r = ds.iloc[0]

    assert r.data == [{'name': 'alice', 'age': 25, 'score': 88.5}]

def test_iloc_with_selected_rows_all_columns():
    r = ds.iloc[1:3]

    assert r.data == [{'name': 'bob', 'age': 30, 'score': 92.0},
        {'name': 'charlie', 'age': 35, 'score': 95.0}]
    
def test_iloc_with_all_rows_first_column():
    r = ds.iloc[:,0]

    assert r.data ==  [{'name': 'alice'}, {'name': 'bob'}, {'name': 'charlie'}] 

def test_iloc_with_specific_rows_and_specific_column():
    r = ds.iloc[[0, 2], [1]]

    assert r.data == [{'age': 25}, {'age': 35}]  

def test_iloc_with_all_rows_and_specific_columns():
    r = ds.iloc[:, [0, 1]]

    assert r.data == [{'name': 'alice', 'age': 25},
        {'name': 'bob', 'age': 30},
        {'name': 'charlie', 'age': 35}]  

def test_iloc_with_all_records_and_tuple_columns():
    r = ds.iloc[:, (0,1)]         

    assert r.data == [{'name': 'alice', 'age': 25},
        {'name': 'bob', 'age': 30},
        {'name': 'charlie', 'age': 35}] 
    

# loc tests

def test_loc_with_single_row():
    r = ds.loc['a']

    assert r.data == [{'name': 'alice', 'age': 25, 'score': 88.5}]

def test_loc_with_selected_rows():
    r = ds.loc[['a', 'c']]

    assert r.data == [{'name': 'alice', 'age': 25, 'score': 88.5},
        {'name': 'charlie', 'age': 35, 'score': 95.0}] 

def test_loc_with_selected_column_from_selected_row():
    r = ds.loc['b', 'score']

    assert r.data ==  [{'score': 92.0}]  

def test_loc_with_range_or_rows():
    r = ds.loc['a': 'c']

    assert r.data ==  [{'name': 'alice', 'age': 25, 'score': 88.5},
        {'name': 'bob', 'age': 30, 'score': 92.0},
        {'name': 'charlie', 'age': 35, 'score': 95.0}]    

def test_loc_with_specific_rows_and_specific_columns():
    r = ds.loc[['a', 'b'], ['name', 'score']]

    assert r.data == [{'name': 'alice', 'score': 88.5}, {'name': 'bob', 'score': 92.0}]      

def test_loc_with_boolean_mask():
      mask = [True, False, False]
      r = ds.loc[mask]

      assert r.data == [{'name': 'alice', 'age': 25, 'score': 88.5}]

def test_loc_with_None_for_column_filter():
    # all_rows = slice(None)
    # col_filter = None
    # r = ds.loc((all_rows, col_filter))

    # assert r.columns == ds.columns
    # assert r.data == ds.data
    r = ds.loc[['a'], None]

    assert r.data == [{'name': 'alice', 'age': 25, 'score': 88.5}]
    assert r.columns == ds.columns



