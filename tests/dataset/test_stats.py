from atrax import Dataset
import pytest
from datetime import datetime

def test_head():
    ds = Dataset([
        {'a': 1},
        {'a': 2},
        {'a': 3}
    ])

    head = ds.head()
    assert head.shape() == (3, 1)

def test_head_with_n_rows():
    ds = Dataset([
        {'a': 1},
        {'a': 2},
        {'a': 3},
        {'a': 4},
        {'a': 5}
    ])   
    head = ds.head(2)
    assert head.shape() == (2, 1) 

def test_tail():
    ds = Dataset([
        {'a': 1},
        {'a': 2},
        {'a': 3}
    ])

    tail = ds.tail()
    assert tail.shape() == (3, 1) 

def test_tail_with_n_rows():
    ds = Dataset([
        {'a': 1},
        {'a': 2},
        {'a': 3},
        {'a': 4},
        {'a': 5}
    ])   
    tail = ds.tail(2)
    assert tail.shape() == (2, 1)   

def test_describe():
    ds = Dataset([{'x': 1}, {'x': 3}, {'x': 5}])
    summary = ds.describe()
    assert summary.data == [{'stat': 'mean', 'x': 3},
                {'stat': 'std', 'x': 2.0},
                {'stat': 'min', 'x': 1},
                {'stat': 'Q1', 'x': 1},
                {'stat': 'median', 'x': 3},
                {'stat': 'Q3', 'x': 5},
                {'stat': 'max', 'x': 5},
                {'stat': 'count', 'x': 3}]
    
def test_describe_with_numeric_only():
    ds = Dataset([
        {
            'id': 1,
            'name': 'sally'
        },
        {
            'id': 2,
            'name': 'billy'
        }
    ])
    summary = ds.describe(numeric_only=True)
    assert summary.data == [{'stat': 'mean', 'id': 1.5},
        {'stat': 'std', 'id': 0.71},
        {'stat': 'min', 'id': 1},
        {'stat': 'Q1', 'id': 1},
        {'stat': 'median', 'id': 1.5},
        {'stat': 'Q3', 'id': 2},
        {'stat': 'max', 'id': 2},
        {'stat': 'count', 'id': 2}]
    
def test_info(capsys):
    ds = Dataset([
        {'a': 1, 'b': 'hello'},
        {'a': 2, 'b': 'world'},
        {'a': None, 'b': '!'}
    ])

    ds.info()
    captured = capsys.readouterr().out
    assert 'Data columns (total 2 columns):' in captured
    assert 'Range Index: 3 entries' in captured

def test_info_with_blank_dataset(capsys):
    ds = Dataset([])
    ds.info()
    captured = capsys.readouterr().out

    assert 'No data available' in captured

def test_info_with_str(capsys):
    ds = Dataset([
        {
            'id': 1,
            'name': 'mike',
            'successfull': False,
            'died': datetime(2026, 1, 1),
            'max_pee_amount_in_gallons': 12.25,
            'none_type': None
        }
    ]) 
    ds.info()
    captured = capsys.readouterr().out

    assert 'int' in captured
    assert 'float' in captured
    assert 'bool' in captured
    assert 'datetime' in captured
    assert 'str' in captured   