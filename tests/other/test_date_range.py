import pytest
from datetime import datetime
from atrax import date_range

# Assuming date_range is already defined or imported

def test_date_range_with_end_and_default_freq():
    result = date_range("2023-01-01", "2023-01-03")
    expected = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]
    assert result == expected

def test_date_range_with_periods():
    result = date_range("2023-01-01", periods=3)
    expected = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]
    assert result == expected

def test_date_range_hourly():
    result = date_range("2023-01-01T00:00:00", "2023-01-01T03:00:00", freq='H')
    expected = [
        datetime(2023, 1, 1, 0),
        datetime(2023, 1, 1, 1),
        datetime(2023, 1, 1, 2),
        datetime(2023, 1, 1, 3),
    ]
    assert result == expected

def test_date_range_minute_aliases():
    res1 = date_range("2023-01-01T00:00:00", periods=3, freq='T')
    res2 = date_range("2023-01-01T00:00:00", periods=3, freq='min')
    assert res1 == res2

def test_date_range_with_datetime_inputs():
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 3)
    result = date_range(start, end)
    expected = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]
    assert result == expected

def test_date_range_with_custom_format():
    result = date_range("01-01-2023", "03-01-2023", fmt="%d-%m-%Y")
    expected = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]
    assert result == expected

# ---- Error Cases ----

def test_date_range_raises_on_missing_end_and_periods():
    with pytest.raises(ValueError, match="Either 'end' or 'periods' must be specified."):
        date_range("2023-01-01")

def test_date_range_invalid_freq():
    with pytest.raises(ValueError, match="Unsupported frequency: W"):
        date_range("2023-01-01", periods=3, freq='W')

def test_date_range_negative_periods():
    with pytest.raises(ValueError, match="'periods' must be a positive integer."):
        date_range("2023-01-01", periods=0)

def test_date_range_start_after_end():
    with pytest.raises(ValueError, match="'start' must be before 'end'."):
        date_range("2023-01-05", "2023-01-01")
