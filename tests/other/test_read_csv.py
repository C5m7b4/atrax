import pytest
from io import StringIO
from datetime import datetime
from atrax import Atrax as tx

CSV_DATA = """id,name,sales,date
1,Alice,100.5,2024-07-01
2,Bob,200.0,2024-07-02
3,Charlie,150.25,2024-07-03
"""

def test_read_csv_from_string():
    ds = tx.read_csv(CSV_DATA, from_string=True)
    assert ds.shape() == (3, 4)
    assert ds[0]['id'] == 1
    assert ds[1]['sales'] == 200.0
    assert ds[2]['name'] == 'Charlie'

def test_read_csv_with_usecols():
    ds = tx.read_csv(CSV_DATA, from_string=True, usecols=['id', 'sales'])
    assert ds.shape() == (3, 2)
    assert 'name' not in ds.columns
    assert 'sales' in ds.columns    

def test_read_csv_with_converters():
    conv = {'sales': lambda x: float(x) * 2}
    ds = tx.read_csv(CSV_DATA, from_string=True, converters=conv)
    assert ds[0]['sales'] == 201.0
    assert ds[1]['sales'] == 400.0   

def test_read_csv_with_parse_dates():
    ds = tx.read_csv(CSV_DATA, from_string=True, parse_dates=['date'])
    assert isinstance(ds[0]['date'], datetime)
    assert ds[1]['date'].strftime('%Y-%m-%d') == '2024-07-02'   

def test_read_csv_numeric_fallback():
    malformed_csv = """id,value\n1,abc\n2,3.14\n"""
    ds = tx.read_csv(malformed_csv, from_string=True)
    assert ds[0]['value'] == 'abc'
    assert ds[1]['value'] == 3.14      

def test_read_csv_from_file(tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text(CSV_DATA)
    ds = tx.read_csv(str(file_path))
    assert ds[2]['name'] == 'Charlie'    