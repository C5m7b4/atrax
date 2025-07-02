import pytest
from atrax import Series

def test_shift_positive():
    s = Series([1, 2, 3], name="a", index=["x", "y", "z"])
    shifted = s.shift(1)
    assert shifted.data == [None, 1, 2]
    assert shifted.index == ["x", "y", "z"]

def test_shift_negative():
    s = Series([1, 2, 3], name="a", index=["x", "y", "z"])
    shifted = s.shift(-1)
    assert shifted.data == [2, 3, None]

def test_shift_zero():
    s = Series([1, 2, 3], name="a", index=["x", "y", "z"])
    shifted = s.shift(0)
    assert shifted.data == [1, 2, 3]

def test_shift_with_fill_value():
    s = Series([10, 20, 30])
    shifted = s.shift(2, fill_value=0)
    assert shifted.data == [0, 0, 10]    

def test_shift_negative_beyond_length():
    s = Series([5, 6])
    with pytest.raises(ValueError):
        s.shift(-3)
    

def test_shift_empty_series():
    s = Series([])
    shifted = s.shift(1)
    assert shifted.data == [None]

def test_shift_preserves_name_and_index():
    s = Series([1, 2, 3], name="test", index=["a", "b", "c"])
    shifted = s.shift(1)
    assert shifted.name == "test"
    assert shifted.index == ["a", "b", "c"]

def test_shift_invalid_periods():
    s = Series([1, 2, 3])
    with pytest.raises(TypeError):
        s.shift("1")    