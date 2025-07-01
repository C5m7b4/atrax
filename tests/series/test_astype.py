import pytest
from atrax import Series

def test_astype():
    s = Series([1,2,3])

    c = s.astype('float')

    assert c.data == [1.0, 2.0, 3.0]

def test_astype_with_invalid_type():
    s = Series([1, 2, 3])

    with pytest.raises(ValueError):
        r = s.astype('foo')  

def test_astype_with_primitive():
    s = Series([1,2,3])
    r = s.astype(float)

    assert r.data == [1.0, 2.0, 3.0]   

def test_astype_with_unconvertable():
    s = Series(['cata', 'dog'])

    r = s.astype('int')

    assert r.data == [None, None]          