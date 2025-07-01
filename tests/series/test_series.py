import pytest
from datetime import datetime
from atrax import Series


# creation tests
def test_series_creation_with_index():
    s = Series([10, 20, 30],name='test', index=['a', 'b', 'c'])
    assert s.data == [10, 20, 30]
    assert s.index == ['a', 'b', 'c']
    assert s.name == 'test'

def test_series_creation_without_index():
    s = Series([1,2,3])
    assert s.index == [0, 1, 2]
    assert s.name == ""

def test_series_creation_invalid_index_length():
    with pytest.raises(ValueError):
        Series([1,2,3], index=['a', 'b'])

# repr tests
def test_repr_contains_dtype_and_name():
    s = Series([1,2,3], name='test', index=['a', 'b', 'c'])
    r = repr(s)
    assert 'dtype' in r
    assert 'test' in r
    assert '<Series' in r  

def test_repr_contains_dot_dot_dot():
    s = Series([1,2,3,4,5,6,7,8,9,10, 11])
    r = repr(s)

    assert '...' in r 

def test_repr_html():
    s = Series([1,2,3,4], name='test', index=['a', 'b', 'c', 'd'])
    html = s._repr_html_()

    assert isinstance(html, str) 
    assert '<table' in html
    assert 'Name: test' in html
    assert 'a' in html and '1' in html
    assert 'b' in html and '2' in html
    assert 'c' in html and '3' in html
    assert 'd' in html and '4' in html
    assert '</table>' in html

def test_repr_html_extras():
    s = Series([1,2,3,4,5,6,7,8,9,10,11])
    html = s._repr_html_()

    assert '...' in html

# dtype tests
@pytest.mark.parametrize("data, expected_dtype", [
    ([1.0, 2.0, 3.0], 'float64'),
    ([1,2,3], 'int64'),
    (['a', 'b', 'c'], 'str'),
    ([datetime(2020, 1, 1), datetime(2021, 1, 1)], 'datetime'),
    ([1, 'a', 3.0], 'object')
])

def test_dtype_inference(data, expected_dtype):
    s = Series(data)
    assert s.dtype == expected_dtype

# head and tail
def test_head_and_tail():
    s = Series([1,2,3,4,5,6,7,8], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])  
    h = s.head(3)
    t = s.tail(3)

    assert h.data == [1,2,3]
    assert h.index == ['a', 'b', 'c']  

    assert t.data == [6,7,8]
    assert t.index == ['f', 'g', 'h']


