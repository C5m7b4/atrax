import pytest
from atrax import Series

def test_gt():
    s = Series([1,2,3,4,5,6])

    r1 = s > 4
    assert r1.data == [False, False, False, False, True, True]

def test_lt():
    s = Series([1,2,3,4,5,6])

    r1 = s < 4
    assert r1.data == [True, True, True, False, False, False]        

def test_ge():
    s = Series([1,2,3,4,5,6])

    r1 = s >= 4
    assert r1.data == [False, False, False, True, True, True]    

def test_le():
    s = Series([1,2,3,4,5,6])

    r1 = s <= 4
    assert r1.data == [True, True, True, True, False, False]       

def test_eq():
    s = Series([1,2,3,4,5,6])

    r1 = s == 4
    assert r1.data == [False, False, False, True, False, False]   

def test_ne():
    s = Series([1,2,3,4,5,6])

    r1 = s != 4
    assert r1.data == [True, True, True, False, True, True]                             