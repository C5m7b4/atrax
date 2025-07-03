import pytest
from atrax import Dataset
from atrax.Dataset.group import GroupBy


@pytest.fixture
def dataset():
    return Dataset([
        {'store': 'A', 'sales': 100, 'returns': 5},
        {'store': 'A', 'sales': 150, 'returns': 3},
        {'store': 'B', 'sales': 200, 'returns': 2},
        {'store': 'B', 'sales': 100, 'returns': 4},
        {'store': 'B', 'sales': 150, 'returns': 1},        
    ])

@pytest.fixture
def grouped(dataset):
    return dataset.groupby('store')

def test_sum(grouped):
    result = grouped.sum().data
    assert result[0]['sales_sum'] == 250
    assert result[1]['sales_sum'] == 450

def test_sum_sorted(grouped):
    result = grouped.sum().data
    a = result[0]
    b = result[1]
    assert a['sales_sum'] == 250
    assert a['returns_sum'] == 8
    assert b['sales_sum'] == 450
    assert b['returns_sum'] == 7        

def test_mean(grouped):
    result = grouped.mean().data
    a = result[0]
    b = result[1]
    assert a['sales_mean'] == 125
    assert b['sales_mean'] == 150

def test_mean(grouped):
    result = grouped.avg().data
    a = result[0]
    b = result[1]
    assert a['sales_avg'] == 125
    assert b['sales_avg'] == 150    

def test_min(grouped):
    result = grouped.min().data
    a = result[0]
    b = result[1]
    assert a['sales_min'] == 100
    assert b['sales_min'] == 100  

def test_max(grouped):
    result = grouped.max().data
    a = result[0]
    b = result[1]
    assert a['sales_max'] == 150
    assert b['sales_max'] == 200 

def test_count(grouped):
    result = grouped.count().data
    a = result[0]
    b = result[1]
    assert a['sales_count'] == 2
    assert b['sales_count'] == 3  

def test_first(grouped):
    result = grouped.first().data
    a = result[0]
    b = result[1]
    assert a['sales_first'] == 100
    assert b['sales_first'] == 200    

def test_last(grouped):
    result = grouped.last().data
    a = result[0]
    b = result[1]
    assert a['sales_last'] == 150
    assert b['sales_last'] == 150 

def test_size(grouped):
    result = grouped.size().data
    a = result[0]
    b = result[1]
    assert a['size'] == 2
    assert b['size'] == 3   

def test_agg_named(grouped):
    result = grouped.agg(sales_total=('sales', 'sum'), return_avg=('returns', 'mean')).data
    a = result[0]
    b = result[1]
    assert a['sales_total'] == 250
    assert round(b['return_avg'], 2) == 2.33   

def test_agg_dict(grouped):
    result = grouped.agg({'sales': ['sum', 'mean']}).data
    a = result[0]
    assert 'sales_sum' in a
    assert 'sales_mean' in a       

def test_apply(grouped):
    def custom(rows):
        return {'double_sales': sum(r['sales'] for r in rows) * 2}

    result = grouped.apply(custom).data
    a = next(r for r in result if r['store'] == 'A')
    b = next(r for r in result if r['store'] == 'B')
    assert a['double_sales'] == 500
    assert b['double_sales'] == 900    

def test_transform(grouped):
    def add_bonus(rows):
        return [{'sales': r['sales'], 'bonus': r['sales'] * 0.1} for r in rows]

    result = grouped.transform(add_bonus).data
    for r in result:
        assert 'bonus' in r
        assert r['bonus'] == r['sales'] * 0.1 

def test_filter(grouped):
    result = grouped.filter(lambda rows: sum(r['sales'] for r in rows) > 300).data
    stores = {r['store'] for r in result}
    assert 'B' in stores
    assert 'A' not in stores  

def test_describe(grouped):
    result = grouped.describe().data
    for r in result:
        assert 'sales_mean' in r
        assert 'returns_max' in r                      


def test_cumsum(grouped):
    result = grouped.cumsum().data
    a_sums = [r['sales_cumsum'] for r in result if r['store'] == 'A']
    b_sums = [r['sales_cumsum'] for r in result if r['store'] == 'B']
    assert a_sums == [100, 250]
    assert b_sums == [200, 300, 450]


def test_cumcount(grouped):
    result = grouped.cumcount().data
    a_counts = [r['cumcount'] for r in result if r['store'] == 'A']
    b_counts = [r['cumcount'] for r in result if r['store'] == 'B']
    assert a_counts == [0, 1]
    assert b_counts == [0, 1, 2]


def test_rank(grouped):
    result = grouped.rank().data
    a_ranks = [r['sales_rank'] for r in result if r['store'] == 'A']
    b_ranks = [r['sales_rank'] for r in result if r['store'] == 'B']
    assert a_ranks == [1.0, 2.0]
    assert b_ranks == [3.0, 1.0, 2.0]

# new tests written on 7/3/2025 for item access in groupby
# 
data = [
    {'store_number': 1, 'total_sales': 100, 'items_sold': 10},
    {'store_number': 1, 'total_sales': 150, 'items_sold': 15},
    {'store_number': 2, 'total_sales': 200, 'items_sold': 20},
    {'store_number': 2, 'total_sales': 50,  'items_sold': 5}
]  

def test_groupby_getitem_sum():
    ds = Dataset(data)
    grouped = ds.groupby("store_number")['total_sales'].sum()

    assert grouped[1] == 250
    assert grouped[2] == 250

def test_groupby_getitem_mean():
    ds = Dataset(data)
    grouped = ds.groupby('store_number')['total_sales'].agg('mean')

    assert grouped[1] == 125
    assert grouped[2] == 125

def test_groupby_getitem_custom_func():
    ds = Dataset(data)
    grouped = ds.groupby('store_number')['total_sales'].agg(lambda x: max(x) - min(x))

    assert grouped[1] == 50
    assert grouped[2] == 150  

def test_groupby_getitem_transform():
    ds = Dataset(data)
    grouped = ds.groupby('store_number')['total_sales'].transform(lambda x: [val / sum(x) for val in x])    

    assert grouped.data[0]['total_sales'] == 0.4
    assert grouped.data[1]['total_sales'] == 0.6

def test_groupby_getitem_invalid_column():    
    try:
        ds = Dataset(data)
        ds.groupby('store_number')['some_column']
        assert False
    except KeyError:
        assert True
       