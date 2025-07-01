from atrax import Dataset, Series
import pytest
from test_data import data
from datetime import datetime


def make_dataset():
    """Create a simple test dataset for testing."""
    return Dataset([
        {'name': 'Apple', 'type': 'fruit', 'price': 1.2, 'id': 100},
        {'name': 'Broccoli', 'type': 'veggie', 'price': 2.5, 'id': 101},
        {'name': 'Carrot', 'type': 'veggie', 'price': 1.0, 'id': 102},
    ])


def make_empty_dataset():
    """Create an empty dataset for edge case testing."""
    return Dataset([])


class TestSetIndexWithColumnName:
    """Test set_index using a string column name."""
    
    def test_set_index_with_column_inplace_no_drop(self):
        """Test setting index with column name, inplace=True, drop=False."""
        ds = make_dataset()
        original_data = ds.data.copy()
        
        result = ds.set_index('name', inplace=True, drop=False)
        
        # Should return None when inplace=True
        assert result is None
        
        # Index should be set to the column values
        assert ds._index == ['Apple', 'Broccoli', 'Carrot']
        assert ds._index_name == 'name'
        
        # Data should remain unchanged when drop=False
        assert ds.data == original_data
        assert 'name' in ds.columns
    
    def test_set_index_with_column_inplace_with_drop(self):
        """Test setting index with column name, inplace=True, drop=True."""
        ds = make_dataset()
        
        result = ds.set_index('name', inplace=True, drop=True)
        
        # Should return None when inplace=True
        assert result is None
        
        # Index should be set to the column values
        assert ds._index == ['Apple', 'Broccoli', 'Carrot']
        assert ds._index_name == 'name'
        
        # Column should be dropped from data
        assert 'name' not in ds.columns
        assert ds.columns == ['type', 'price', 'id']
        for row in ds.data:
            assert 'name' not in row
    
    def test_set_index_with_column_not_inplace_no_drop(self):
        """Test setting index with column name, inplace=False, drop=False."""
        ds = make_dataset()
        original_data = ds.data.copy()
        original_index = ds._index.copy()
        
        new_ds = ds.set_index('price', inplace=False, drop=False)
        
        # Should return new Dataset when inplace=False
        assert new_ds is not None
        assert new_ds is not ds
        
        # Original dataset should remain unchanged
        assert ds._index == original_index
        assert ds.data == original_data
        assert 'price' in ds.columns
        
        # New dataset should have updated index
        assert new_ds._index == [1.2, 2.5, 1.0]
        assert new_ds._index_name == 'price'
        assert 'price' in new_ds.columns  # should not be dropped
    
    def test_set_index_with_column_not_inplace_with_drop(self):
        """Test setting index with column name, inplace=False, drop=True."""
        ds = make_dataset()
        original_data = ds.data.copy()
        
        new_ds = ds.set_index('id', inplace=False, drop=True)
        
        # Should return new Dataset when inplace=False
        assert new_ds is not None
        assert new_ds is not ds
        
        # Original dataset should remain unchanged
        assert ds.data == original_data
        assert 'id' in ds.columns
        
        # New dataset should have updated index and dropped column
        assert new_ds._index == [100, 101, 102]
        assert new_ds._index_name == 'id'
        assert 'id' not in new_ds.columns
        assert new_ds.columns == ['name', 'type', 'price']


class TestSetIndexWithCallable:
    """Test set_index using a callable function."""
    
    def test_set_index_with_callable_inplace(self):
        """Test setting index with callable, inplace=True."""
        ds = make_dataset()
        
        # Use a function that extracts the first letter of name
        result = ds.set_index(lambda row: row['name'][0], inplace=True)
        
        assert result is None
        assert ds._index == ['A', 'B', 'C']
        assert ds._index_name is None  # Anonymous index
        
        # All columns should remain (nothing to drop)
        assert ds.columns == ['name', 'type', 'price', 'id']
    
    def test_set_index_with_callable_not_inplace(self):
        """Test setting index with callable, inplace=False."""
        ds = make_dataset()
        original_data = ds.data.copy()
        
        # Use a function that computes price * 10
        new_ds = ds.set_index(lambda row: row['price'] * 10, inplace=False)
        
        assert new_ds is not None
        assert new_ds is not ds
        
        # Original unchanged
        assert ds.data == original_data
        
        # New dataset has computed index
        assert new_ds._index == [12.0, 25.0, 10.0]
        assert new_ds._index_name is None
        assert new_ds.columns == ['name', 'type', 'price', 'id']
    
    def test_set_index_with_callable_complex_function(self):
        """Test setting index with a more complex callable."""
        ds = make_dataset()
        
        # Function that combines multiple fields
        def make_index(row):
            return f"{row['type'][:1].upper()}-{row['id']}"
        
        ds.set_index(make_index, inplace=True)
        
        assert ds._index == ['F-100', 'V-101', 'V-102']
        assert ds._index_name is None


class TestSetIndexWithSequence:
    """Test set_index using explicit sequence of values."""
    
    def test_set_index_with_list_inplace(self):
        """Test setting index with list of values, inplace=True."""
        ds = make_dataset()
        custom_index = ['item1', 'item2', 'item3']
        
        result = ds.set_index(custom_index, inplace=True)
        
        assert result is None
        assert ds._index == custom_index
        assert ds._index_name is None
        
        # All columns should remain
        assert ds.columns == ['name', 'type', 'price', 'id']
    
    def test_set_index_with_tuple_not_inplace(self):
        """Test setting index with tuple of values, inplace=False."""
        ds = make_dataset()
        custom_index = (10, 20, 30)
        
        new_ds = ds.set_index(custom_index, inplace=False)
        
        assert new_ds is not None
        assert new_ds is not ds
        assert new_ds._index == [10, 20, 30]
        assert new_ds._index_name is None
    
    def test_set_index_with_series(self):
        """Test setting index with Series object."""
        ds = make_dataset()
        index_series = Series(['x', 'y', 'z'])
        
        ds.set_index(index_series, inplace=True)
        
        assert ds._index == ['x', 'y', 'z']
        assert ds._index_name is None
    
    def test_set_index_with_datetime_values(self):
        """Test setting index with datetime values."""
        ds = make_dataset()
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)
        ]
        
        ds.set_index(dates, inplace=True)
        
        assert ds._index == dates
        assert ds._index_name is None


class TestSetIndexErrorCases:
    """Test error cases and invalid inputs."""
    
    def test_set_index_invalid_column_name(self):
        """Test error when column doesn't exist."""
        ds = make_dataset()
        
        with pytest.raises(KeyError) as exc_info:
            ds.set_index('nonexistent_column')
        
        assert "Column 'nonexistent_column' not found in dataset" in str(exc_info.value)
    
    def test_set_index_sequence_wrong_length(self):
        """Test error when sequence length doesn't match dataset."""
        ds = make_dataset()
        
        # Too short
        with pytest.raises(ValueError) as exc_info:
            ds.set_index(['a', 'b'])  # only 2 items for 3 rows
        
        assert "Index length must match number of rows" in str(exc_info.value)
        
        # Too long
        with pytest.raises(ValueError) as exc_info:
            ds.set_index(['a', 'b', 'c', 'd'])  # 4 items for 3 rows
        
        assert "Index length must match number of rows" in str(exc_info.value)
    
    def test_set_index_invalid_type(self):
        """Test error with invalid column type."""
        ds = make_dataset()
        
        with pytest.raises(TypeError) as exc_info:
            ds.set_index(123)  # integer is not valid
        
        assert "`column` must be a string, callable, or sequence of index values" in str(exc_info.value)
        
        with pytest.raises(TypeError) as exc_info:
            ds.set_index({'not': 'valid'})  # dict is not valid
        
        assert "`column` must be a string, callable, or sequence of index values" in str(exc_info.value)


class TestSetIndexEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_set_index_empty_dataset(self):
        """Test setting index on empty dataset."""
        ds = make_empty_dataset()
        
        # Should work with empty sequence
        ds.set_index([], inplace=True)
        assert ds._index == []
        assert ds._index_name is None
        
        # Should fail with non-empty sequence
        with pytest.raises(ValueError):
            ds.set_index(['a'], inplace=True)
    
    def test_set_index_callable_with_empty_dataset(self):
        """Test setting index with callable on empty dataset."""
        ds = make_empty_dataset()
        
        # Should work even though callable won't be called
        ds.set_index(lambda x: x['name'], inplace=True)
        assert ds._index == []
        assert ds._index_name is None
    
    def test_set_index_preserves_other_attributes(self):
        """Test that set_index preserves other dataset attributes."""
        ds = make_dataset()
        original_columns = ds.columns.copy()
        
        # Set initial index for comparison
        ds._index = ['old1', 'old2', 'old3']
        ds._index_name = 'old_index'
        
        # Set new index
        ds.set_index('name', inplace=True, drop=False)
        
        # Check that columns are preserved (except when dropping)
        assert ds.columns == original_columns
    
    def test_set_index_with_none_values(self):
        """Test setting index with sequence containing None values."""
        ds = make_dataset()
        
        index_with_none = ['a', None, 'c']
        ds.set_index(index_with_none, inplace=True)
        
        assert ds._index == ['a', None, 'c']
    
    def test_set_index_with_duplicate_values(self):
        """Test setting index with duplicate values (should be allowed)."""
        ds = make_dataset()
        
        duplicate_index = ['same', 'same', 'different']
        ds.set_index(duplicate_index, inplace=True)
        
        assert ds._index == ['same', 'same', 'different']
    
    def test_set_index_drop_parameter_ignored_for_non_string(self):
        """Test that drop parameter is ignored when not using string column."""
        ds = make_dataset()
        original_columns = ds.columns.copy()
        
        # drop=True should be ignored for callable
        ds.set_index(lambda row: row['name'], inplace=True, drop=True)
        assert ds.columns == original_columns
        
        # drop=True should be ignored for sequence
        ds = make_dataset()
        ds.set_index(['x', 'y', 'z'], inplace=True, drop=True)
        assert ds.columns == original_columns


class TestSetIndexIntegration:
    """Integration tests with real data."""
    
    def test_set_index_with_test_data(self):
        """Test set_index with the imported test data."""
        ds = Dataset(data[:3])  # Use first 3 items from test_data
        
        # Test with product_code column
        result = ds.set_index('product_code', inplace=False, drop=True)
        
        assert result._index == [
            '0002663380147',
            '0002663358001', 
            '0003644920475'
        ]
        assert result._index_name == 'product_code'
        assert 'product_code' not in result.columns
        assert 'description' in result.columns
    
    def test_set_index_chaining_compatible(self):
        """Test that set_index works in method chaining scenarios."""
        ds = make_dataset()
        
        # Since inplace=True returns None, we can't chain
        # But inplace=False should work for chaining
        result = ds.set_index('name', inplace=False, drop=True)
        
        # Should be able to access the new dataset
        assert len(result.data) == 3
        assert result._index == ['Apple', 'Broccoli', 'Carrot']
        
        # Could theoretically chain more operations on result
        subset = result[['type', 'price']]
        assert len(subset.data) == 3
