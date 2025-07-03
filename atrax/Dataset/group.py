from collections import defaultdict


class GroupBy:
    """
    GroupBy class for aggregating data based on specified keys.

    Supports:
    - agg (with named or dict-style aggregations)
    - sum, mean, min, max, count, first, last, size
    - apply (custom row-level logic)
    """

    def __init__(self, data, by, sort=False):
        self.by = by
        self._by_list = by if isinstance(by, list) else [by]
        self.data = data
        self.sort = sort
        self.groups = self._group_data()
        self._selected_column = None

    def __getitem__(self, key):
        """
            Allow column access after groupby, e.g. ds.groupby('col')['target']

            Parameters:
            ----------
                key (str): column to select
            
            Returns:
                GroupBy
        """
        if not isinstance(key, str):
            raise TypeError(f"Unsupported key type: {type(key)}")
        
        column_found = any(key in row for rows in self.groups.values() for row in rows)
        if not column_found:
            raise KeyError(f"Columns '{key}' not found in grouped data")
        

        new_groups = {}
        for group_key, rows in self.groups.items():
            new_groups[group_key] = [row[key] for row in rows if key in row]
        
        # create a new groupby object with same grouping keys and filtered groups
        new_obj = GroupBy(self.data, self._by_list)
        new_obj.groups = new_groups
        new_obj._selected_column = key
        return new_obj


    def _group_data(self):
        grouped = defaultdict(list)
        is_single_key = len(self._by_list) == 1


        for row in self.data:
            if is_single_key:
                key = row[self._by_list[0]]
            else:
                key = tuple(row[k] for k in self._by_list)
            grouped[key].append(row)

        if self.sort:
            return dict(sorted(grouped.items()))
        return grouped

    def _aggregate(self, agg_func_map, named_agg):
        from .dataset import Dataset
        result = []
        for group_key, rows in self.groups.items():
            col_data = defaultdict(list)
            for row in rows:
                for col, val in row.items():
                    col_data[col].append(val)

            aggregated_row = {}

            if named_agg:
                # named agg: e.g. sales_mean=('sales', 'mean')
                for output_col, (input_col, agg_func) in agg_func_map.items():
                    values = col_data.get(input_col, [])
                    aggregated_row[output_col] = self._apply_func(agg_func, values, input_col)
            else:
                # dict agg: e.g. sales: ['sum', 'mean']
                for input_col, agg_funcs in agg_func_map.items():
                    if not isinstance(agg_funcs, list):
                        agg_funcs = [agg_funcs]
                    values = col_data.get(input_col, [])
                    for agg_func in agg_funcs:
                        result_col = self._output_col_name(input_col, agg_func)
                        aggregated_row[result_col] = self._apply_func(agg_func, values, input_col)

            for i, col in enumerate(self._by_list):
                aggregated_row[col] = group_key[i]

            result.append(aggregated_row)
        return Dataset(result)

    def _apply_func(self, func, values, input_col=None):
        if isinstance(func, str):
            if func == 'sum':
                return sum(values)
            elif func == 'mean' or func == 'avg':
                return sum(values) / len(values) if values else 0
            elif func == 'count':
                return len(values)
            elif func == 'min':
                return min(values) if values else None
            elif func == 'max':
                return max(values) if values else None
            elif func == 'first':
                return values[0] if values else None
            elif func == 'last':
                return values[-1] if values else None
            else:
                raise ValueError(f"Unknown aggregation function: {func}")
        elif callable(func):
            return func(values)
        else:
            raise TypeError(f"Aggregation function must be a string or callable, got {type(func)}")

    def _output_col_name(self, input_col, func):
        if isinstance(func, str):
            return f"{input_col}_{func}"
        elif callable(func):
            return f"{input_col}_{func.__name__}"
        else:
            raise TypeError("Aggregation function must be a string or callable")

    def agg(self, func=None, **kwargs):
        from atrax import Dataset
        if self._selected_column is not None:
            # 🟦 Single-column case: ds.groupby(...)[col].agg('sum') or agg(func)
            if func is None:
                raise ValueError("Must provide an aggregation function for selected column")

            result = {}
            for group_key, values in self.groups.items():
                if callable(func):
                    result[group_key] = func(values)
                elif isinstance(func, str):
                    result[group_key] = self._apply_named_func(values, func)
                else:
                    raise TypeError(f"Invalid aggregation function: {func}")
            return result

        else:
            # 🟩 Multi-column case: ds.groupby(...).agg({'col1': 'sum', ...})
            agg_map = {}
            if isinstance(func, dict):
                return self._aggregate(func, named_agg=False)
            elif kwargs:
                if all(isinstance(v, tuple) and len(v) == 2 for v in kwargs.values()):
                    return self._aggregate(kwargs, named_agg=True)
                else:
                    raise TypeError("Named aggregation must be in the form output_col=(input_col, agg_func)")
            else:
                raise ValueError("agg() requires a dict or named arguments when no column is selected")

            # result = []
            # for group_key, rows in self.groups.items():
            #     agg_row = {}
            #     if len(self._by_list) == 1:
            #         agg_row[self.by[0]] = group_key
            #     else:
            #         for i, col in enumerate(self._by_list):
            #             agg_row[col] = group_key=[i]
            #     for col, agg_func in agg_map.items():
            #         values = [row[col] for row in rows if col in row]
            #         if callable(agg_func):
            #             agg_value = agg_func(values)
            #         elif isinstance(agg_func, str):
            #             agg_value = self._apply_named_func(values, agg_func)
            #         else:
            #             raise TypeError(f"Unsupported agg function: {agg_func}")
            #         agg_row[f"{col}_{agg_func}"] = agg_value
            #     result.append(agg_row)
            # return Dataset(result)
  

    def sum(self):
        if self._selected_column:
            return self.agg('sum')
        return self.agg({col: 'sum' for col in self._numeric_columns()})

    def mean(self):
        return self.agg({col: 'mean' for col in self._numeric_columns()})

    def avg(self):
        return self.agg({col: 'avg' for col in self._numeric_columns()})

    def min(self):
        return self.agg({col: 'min' for col in self._numeric_columns()})

    def max(self):
        return self.agg({col: 'max' for col in self._numeric_columns()})

    def count(self):
        return self.agg({col: 'count' for col in self.data[0].keys()})

    def first(self):
        return self.agg({col: 'first' for col in self.data[0].keys()})

    def last(self):
        return self.agg({col: 'last' for col in self.data[0].keys()})

    def size(self):
        from .dataset import Dataset
        result = []
        for group_key, rows in self.groups.items():
            row = {col: key for col, key in zip(self._by_list, group_key)}
            row['size'] = len(rows)
            result.append(row)
        return Dataset(result)

    def apply(self, func):
        """
        Apply a custom function to each group.
        The function should return a dict or list of dicts.
        """
        from .dataset import Dataset
        result = []
        for group_key, rows in self.groups.items():
            out = func(rows)
            if isinstance(out, dict):
                for i, col in enumerate(self._by_list):
                    out[col] = group_key[i]
                result.append(out)
            elif isinstance(out, list):
                for row in out:
                    for i, col in enumerate(self._by_list):
                        row[col] = group_key[i]
                    result.append(row)
            else:
                raise TypeError("apply() must return dict or list of dicts")
        return Dataset(result)

    def _numeric_columns(self):
        numeric_cols = set()
        for row in self.data:
            for col, val in row.items():
                if isinstance(val, (int, float)):
                    numeric_cols.add(col)
        return list(numeric_cols)
    
    def transform(self, func):
        """
        Apply a function to each group and broadcast result back to row level.
        The function should return a list of dicts with same length as group.
        """
        from .dataset import Dataset
        result = []
        for group_key, rows in self.groups.items():
            out = func(rows)

            if not isinstance(out, list) or len(out) != len(rows):
                raise ValueError("transform() must return a list of dicts equal in length to input group")
            
            for i, val in enumerate(out):
                if isinstance(val, dict):
                    row = val.copy()
                else:
                    row = {self._selected_column or 'value': val}

                # attach group keys
                if len(self._by_list) == 1:
                    row[self._by_list[0]] = group_key
                else:
                    for j, col in enumerate(self._by_list):
                        row[col] = group_key[j]

                result.append(row)


        return Dataset(result) 

    def filter(self, func):
        """
        Keep only the groups for which the function returns True.
        """
        from .dataset import Dataset
        result = []
        for group_key, rows in self.groups.items():
            if func(rows):
                for row in rows:
                    for i, col in enumerate(self._by_list):
                        row[col] = group_key[i]
                    result.append(row)
        return Dataset(result)      
    
    def describe(self):
        from .dataset import Dataset
        stats = ['count', 'mean', 'min', 'max']
        numeric_cols = self._numeric_columns()
        result = []
        for group_key, rows in self.groups.items():
            summary = {}
            for col in numeric_cols:
                values = [row[col] for row in rows if isinstance(row[col], (int, float))]
                summary[f'{col}_count'] = len(values)
                summary[f'{col}_mean'] = sum(values) / len(values) if values else None
                summary[f'{col}_min'] = min(values) if values else None
                summary[f'{col}_max'] = max(values) if values else None
            for i, col in enumerate(self._by_list):
                summary[col] = group_key[i]
            result.append(summary)
        return Dataset(result)    
    
    def cumsum(self):
        from .dataset import Dataset
        result = []
        numeric_cols = self._numeric_columns()
        for group_key, rows in self.groups.items():
            cum_sums = {col: 0 for col in numeric_cols}
            for row in rows:
                new_row = row.copy()
                for col in numeric_cols:
                    cum_sums[col] += row.get(col, 0)
                    new_row[f'{col}_cumsum'] = cum_sums[col]
                for i, col in enumerate(self._by_list):
                    new_row[col] = group_key[i]
                result.append(new_row)
        return Dataset(result)    
    
    def cumcount(self):
        from .dataset import Dataset
        result = []
        for group_key, rows in self.groups.items():
            for i, row in enumerate(rows):
                new_row = row.copy()
                new_row['cumcount'] = i
                for j, col in enumerate(self._by_list):
                    new_row[col] = group_key[j]
                result.append(new_row)
        return Dataset(result)    
    
    def rank(self):
        from .dataset import Dataset
        import numpy as np
        result = []
        numeric_cols = self._numeric_columns()
        for group_key, rows in self.groups.items():
            for col in numeric_cols:
                values = [row[col] for row in rows]
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
                ranks = [0] * len(values)
                i = 0
                while i < len(values):
                    j = i
                    while j + 1 < len(values) and values[sorted_indices[j]] == values[sorted_indices[j + 1]]:
                        j += 1
                    rank_val = (i + j + 2) / 2  # average rank (1-based)
                    for k in range(i, j + 1):
                        ranks[sorted_indices[k]] = rank_val
                    i = j + 1
                for idx, row in enumerate(rows):
                    row[f'{col}_rank'] = ranks[idx]
            for row in rows:
                for i, col in enumerate(self._by_list):
                    row[col] = group_key[i]
                result.append(row)
        return Dataset(result)
    

    def _apply_named_func(self, values, func_name):
        if func_name == 'mean':
            return sum(values) / len(values) if values else None
        elif func_name == 'sum':
            return sum(values)
        elif func_name == 'max':
            return max(values)
        elif func_name == 'min':
            return min(values)
        elif func_name == 'avg':
            return sum(values) / len(values) if values else None
        elif func_name == 'count':
            return len(values)
        elif func_name == 'first':
            return values[0]
        elif func_name == 'last':
            return values[len(values)-1]
        elif func_name == 'size':
            return len(values)
        else:
            raise ValueError(f"Unsupported aggregation function: {func_name}")    

