import pytest
import numpy as np
from atrax import Series

def test_unique():
    s = Series([1, 2, 3, 1, 4, 5, 6, 5])
    u = s.unique()

    assert isinstance(u, np.ndarray)

def test_unique_values():
    s = Series([1, 2, 1, 2, 3])
    r = set(s.unique())

    assert r == {1, 2, 3}

def test_nunique():
    s = Series([1, 2, 1, 2, 3])
    n = s.nunique()

    assert n == 3