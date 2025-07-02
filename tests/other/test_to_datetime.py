import pytest
from datetime import datetime
from atrax import to_datetime  # adjust to actual module path

# If using Series from atrax
from atrax.Series.series import Series

# ---- BASIC CASES ----

def test_single_iso_string():
    assert to_datetime("2023-10-01") == datetime(2023, 10, 1)

def test_single_with_format():
    assert to_datetime("01-10-2023", fmt="%d-%m-%Y") == datetime(2023, 10, 1)

def test_single_datetime_passthrough():
    d = datetime(2023, 10, 1)
    assert to_datetime(d) is d

def test_list_of_iso_strings():
    result = to_datetime(["2023-10-01", "2023-10-02"])
    assert result == [datetime(2023, 10, 1), datetime(2023, 10, 2)]

def test_list_with_format():
    result = to_datetime(["01-10-2023", "02-10-2023"], fmt="%d-%m-%Y")
    assert result == [datetime(2023, 10, 1), datetime(2023, 10, 2)]

# ---- FALLBACK FORMATS ----

@pytest.mark.parametrize("input_str,expected", [
    ("2023-10-01", datetime(2023, 10, 1)),
    ("10/01/2023", datetime(2023, 10, 1)),
    ("01-10-2023", datetime(2023, 1, 10)),
])
def test_fallback_formats(input_str, expected):
    assert to_datetime(input_str) == expected

# ---- ERROR HANDLING ----

def test_invalid_string_raises():
    with pytest.raises(ValueError):
        to_datetime("not-a-date")

def test_invalid_string_coerce():
    assert to_datetime("not-a-date", errors="coerce") is None

def test_list_with_mixed_valid_invalid_coerce():
    result = to_datetime(["2023-10-01", "bad-date"], errors="coerce")
    assert result == [datetime(2023, 10, 1), None]

# ---- NONE / EMPTY ----

@pytest.mark.parametrize("val", [None, "", "NaT"])
def test_null_like_values_coerce(val):
    assert to_datetime(val, errors="coerce") is None

def test_null_like_in_list():
    result = to_datetime(["2023-10-01", None, ""], errors="coerce")
    assert result == [datetime(2023, 10, 1), None, None]

def test_null_like_raises():
    with pytest.raises(TypeError):
        to_datetime(None)

# ---- SERIES SUPPORT ----

def test_series_input_iterable():
    s = Series(["2023-10-01", "2023-10-02"])
    result = to_datetime(s)
    assert result == [datetime(2023, 10, 1), datetime(2023, 10, 2)]

# Optional: for Series with `.data` instead of iterable
def test_series_input_data_property():
    class CustomSeries(Series):
        def __init__(self, data):
            self.data = data

    s = CustomSeries(["2023-10-01", "2023-10-02"])
    result = to_datetime(s)
    assert result == [datetime(2023, 10, 1), datetime(2023, 10, 2)]

# ---- TYPE ERRORS ----

@pytest.mark.parametrize("val", [123, 3.14, {"date": "2023-10-01"}, (1, 2)])
def test_invalid_input_types(val):
    with pytest.raises(TypeError):
        to_datetime(val)
