import pytest
from atrax import Series

def test_and():
    s = Series([1,2,3,4,5,6,7,8,9])

    r1 = ((s > 2) & (s < 8))
    assert r1.data == [False, False, True, True, True, True, True, False, False]

def test_and_without_series():
    s = Series([1,2,3,4,5,6,7,8])

    other = 1
    with pytest.raises(TypeError):
        (s > 1) & other 

def test_series_must_have_same_length():
    s = Series([1,2,3,4,5])
    s1 = Series([1,2,3])

    with pytest.raises(ValueError):
        (s > 1) & (s1 < 3)

def test_or():
    s = Series([1,2,3,4,5,6,7,8,9])

    r1 = (s > 5) | (s == 2)
    assert r1.data == [False, True, False, False, False, True, True, True, True]

def test_or_without_series():
    s = Series([1,2,3,4,5,6,7,8])

    other = 1
    with pytest.raises(TypeError):
        (s > 1) | other  

def test_series_must_have_same_length_with_or():
    s = Series([1,2,3,4,5])
    s1 = Series([1,2,3])

    with pytest.raises(ValueError):
        (s > 1) | (s1 < 3)               