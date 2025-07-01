import pytest
from atrax import Series

def label(x):
    return 'even' if x % 2 == 0 else 'odd'

def test_apply():
    s = Series([1, 2, 3, 4])

    r = s.apply(label)

    assert r.data == ['odd', 'even', 'odd', 'even']



