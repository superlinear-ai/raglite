# AGENTS.md

## Commands

1. Run `uv run poe lint` to lint the project.
2. Run `uv run poe test` to test the project.

## Style rules

### S1: Prefer full words over abbreviations in names
> Prefer full words to promote self-documenting and unambiguous code. Reserve abbreviations for universal terms (api, id, min, max, url, ...).

✅ Good:
```python
max_capacity = 9000
state_index = 0
time_series_component = ...
response_buffer = []

def calculate_average(values: Iterable[float]) -> float: ...
```

❌ Bad:
```python
max_cap = 9000
state_idx = 0
ts_comp = ...
resp_buf = []

def calc_avg(vals: Iterable[float]) -> float: ...
```

### S2: Prefer inlining over indirection
> Prefer inlining logic over unnecessary indirection from intermediate variables or (private) functions that could be avoided. Exceptions: creating return variables, and creating intermediate variables or (private) functions that are referenced at least three times.

✅ Good:
```python
if len(items) > 0 and all(item.is_valid for item in items):
    process_batch(items)

def total(items: Iterable[Item]) -> float:
    return sum(item.price * item.quantity for item in items)
```

❌ Bad:
```python
has_items = len(items) > 0
all_valid = all(item.is_valid for item in items)
if has_items and all_valid:
    process_batch(items)

def total(items: Iterable[Item]) -> float:
    def _item_cost(item: Item) -> float:
        return item.price * item.quantity
    return sum(_item_cost(item) for item in items)
```
