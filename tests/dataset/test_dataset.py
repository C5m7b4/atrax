from atrax import Dataset
import pytest

def test_dataset_from_list_of_dicts():
    data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    ds = Dataset(data)

    assert ds.columns == ['a', 'b']
    assert ds.data == [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]

def test_dataset_from_dict_of_lists():
    data = {'a': [1,3], 'b': [2,4]}
    ds = Dataset(data)

    assert ds.columns == ['a', 'b']
    assert ds.data == [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]

def test_dataset_with_different_column_lengths():
    data = {'a': [1,3], 'b': [2,4,6]}

    with pytest.raises(ValueError):
        ds = Dataset(data)

def test_repr_under_ten_lines():
    ds = Dataset([
        {'a': 1, 'b': 2},
        {'a': 3, 'b': 4}
    ])        
    output = repr(ds)
    expected_lines = [
        'a, b',
        '1, 2',
        '3, 4'
    ]
    for line in expected_lines:
        assert line in output
    
    assert '...' not in output

def test_repr_over_ten_lines():
    data = [{'a': i, 'b': i * 2} for i in range(15)]
    ds = Dataset(data)
    output = repr(ds)

    assert output.startswith("<Dataset />")
    assert '...' in output
    assert '(15) total rows' in output

def test_repr_html_with_empty_dataset():    
    data = []
    ds = Dataset(data)
    html = ds._repr_html_()
    assert html.strip() == '<i>Empty Dataset</i>'

def test_repr_html_basics():
    ds = Dataset([
        {'a': 1, 'b': 2},
        {'a': 3, 'b': 4}
    ])    
    html = ds._repr_html_()

    assert '<table>' in html
    assert '</table>' in html
    assert '<th>a</th>' in html
    assert '<th>b</th>' in html
    assert '<td>1</td>' in html

