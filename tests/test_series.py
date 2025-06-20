from atrax import Atrax

def test_placeholder():
    assert True

def test_series_creation_from_list():
    s = Atrax.Series([1, 2, 3, 4])

    # basic checks
    assert s.data == [1,2,3,4]