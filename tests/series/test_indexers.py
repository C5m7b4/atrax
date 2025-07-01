import pytest
from datetime import datetime
from atrax import Series


# iloc
def test_iloc_access():
    s = Series([1,2,3], index=['a', 'b', 'c'])

    assert s.iloc[0] == 1

def test_iloc_slice():
    s = Series([1,2,3,4,5,6])

    assert s.iloc[0:3].data == [1,2,3]  
    assert s.iloc[1:4].data == [2,3,4]  


# loc
def test_loc_access():
    s = Series([1,2,3], index=['a', 'b', 'c'])

    assert s.loc['a'] == 1  

def test_loc_invalid_end_label():
    s = Series([1,2,3], index=['a', 'b', 'c'])

    with pytest.raises(KeyError):
        s.loc['a': 'z']  

def test_loc_invalid_start_label():
    s = Series([1,2,3], index=['a', 'b', 'c'])

    with pytest.raises(KeyError):
        s.loc['y': 'z']  

def test_loc_slice():
    s = Series([1,2,3,4,5,6,7], index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    assert s.loc['a': 'c'].data == [1,2,3]  

def test_loc_list():
    s = Series([1,2,3,4,5,6,7], index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    assert s.loc[['a', 'c']].data == [1,3]        