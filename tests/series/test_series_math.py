import pytest
from atrax import Series

def test_series_to_array():
    s = Series([1,2,3])
    r = s.to_array()

    assert r == [1,2,3]

def test_series__to_array__():
    s = Series([1,2,3])
    rs = s.__array__()

    r = [int(r) for r in rs]

    assert r == [1,2,3]    

def test_series_mean():
    s = Series([1,2,3,4])   
    r = s.mean()

    assert r == 2.5 

def test_series_std():
    s = Series([1,2,3,4])
    r = round(s.std(),2)

    assert r == 1.29  

def test_series_to_list():
    s = Series([1,2,3,4])
    r = s.to_list()

    assert r == [1,2,3,4]   

def test_series_var():
    s = Series([1,2,3,4])
    r = round(s.var(), 2)

    assert r == 1.67  
    assert s.var(sample=False) == 1.25 

def test_series_var_too_short():
    s = Series([1])

    with pytest.raises(ValueError):
        s.var()

def test_series_sum():
    s = Series([1,2,3])
    r = s.sum()

    assert r == 6   

def test_series_min():
    s = Series([1,2,3])
    r = s.min()

    assert r == 1

def test_series_max():
    s = Series([1,2,3])
    r = s.max()

    assert r == 3   

def test_series_median():
    s = Series([1,2,3,4,5])
    r = s.median()

    assert r == 3 

def test_series_prod():
    s = Series([1,2,3,4])
    r = s.prod()

    assert r ==   24  

def test_series_cumsum():
    s = Series({1,2,3,4})
    r = s.cumsum()

    assert r == [1,3,6,10]  

def test_series_cumprod():
    s = Series([1,2,3,4])
    r = s.cumprod()

    assert r == [1,2,6,24]  

def test_series_cummin():
    s1 = Series([3, 2, 5, 1, 4])
    r = s1.cummin()

    assert r == [3, 2, 2, 1, 1]  

def test_series_cummax():
    s1 = Series([3, 2, 5, 1, 4])
    r = s1.cummax()

    assert r == [3, 3, 5, 5, 5]                  
