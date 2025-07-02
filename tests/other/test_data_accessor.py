import pytest
from datetime import datetime
from atrax import Atrax as tx


@pytest.fixture
def series_with_str_dates():
    return tx.Series(
        ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
        name="sale_date"
    )

@pytest.fixture
def series_with_datetime_objects():
    return tx.Series(
        [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
        name="sale_date"
    )

def test_weekday(series_with_str_dates):
    result = series_with_str_dates.dt.weekday
    expected = [2, 3, 4, 5, 6]  # Wed to Sun
    assert result.data == expected
    assert result.name == "sale_date_weekday"

def test_is_weekend(series_with_str_dates):
    result = series_with_str_dates.dt.is_weekend
    expected = [False, False, False, True, True]
    assert result.data == expected
    assert result.name == "sale_date_is_weekend"  

def test_day(series_with_str_dates):
    result = series_with_str_dates.dt.day
    expected = [1, 2, 3, 4, 5]
    assert result.data == expected
    assert result.name == "sale_date_day"   

def test_month(series_with_str_dates):
    result = series_with_str_dates.dt.month
    expected = [1, 1, 1, 1, 1]
    assert result.data == expected
    assert result.name == "sale_date_month"    

def test_year(series_with_str_dates):
    result = series_with_str_dates.dt.year
    expected = [2025] * 5
    assert result.data == expected
    assert result.name == "sale_date_year"   

def test_datetime_input_equivalence(series_with_datetime_objects):
    str_series = tx.Series(['2025-01-01', '2025-01-02', '2025-01-03'], name="sale_date")
    result_str = str_series.dt.weekday
    result_dt = series_with_datetime_objects.dt.weekday
    assert result_str.data == result_dt.data
    assert result_str.name == result_dt.name   

def test_invalid_type_raises():
    bad_series = tx.Series([123, 456], name="invalid")
    with pytest.raises(TypeError):
        _ = bad_series.dt.day

def test_preserves_index():
    s = tx.Series(['2025-01-01', '2025-01-02'], name="sale_date", index=['a', 'b'])
    result = s.dt.month
    assert result.index == ['a', 'b']
    assert result.data == [1, 1]

def test_name_propagation():
    s = tx.Series(['2025-01-01'], name="my_dates")
    assert s.dt.year.name == "my_dates_year"
    assert s.dt.day.name == "my_dates_day"  

def test_isinstance_date():
    d1 = datetime(2025,1,1)
    d2 = datetime(2025,1,2)
    s = tx.Series([d1, d2])
    assert s.dt.day.data == [1,2]
    assert s.dt.month.data == [1,1]  
    assert s.dt.year.data == [2025, 2025]
