import pytest
from atrax import Series

def test_map_dict():
    s = Series([1,2,3], name='example', index=['a', 'b', 'c'])
    mapping = {1: 'one', 2: 'two', 3: 'three'}
    mapped_s = s.map(mapping)

    assert mapped_s.data == ['one', 'two', 'three']

def test_map_lambda():
    s = Series([1,2,3], name='example', index=['a', 'b', 'c'])

    r = s.map(lambda x: x * 2)

    assert r.data == [2, 4, 6]

def test_map_with_error():
    s = Series([1,2,3])

    with pytest.raises(TypeError):
        s.map('hello')