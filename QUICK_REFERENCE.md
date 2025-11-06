# Quick Reference: StandardScaler Warning Fix

## The Problem
```python
UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
```

## Quick Solutions

### 1. Use FeatureScaler Wrapper (Easiest)
```python
from text-summarizer.utils.preprocessing import FeatureScaler

scaler = FeatureScaler()
scaler.fit(train_dataframe)
scaled = scaler.transform(test_array)  # Works without warning!
```

### 2. Keep Data Types Consistent
```python
# Both as DataFrames
scaler.fit(train_df)
scaler.transform(test_df)  # OK

# Both as arrays
scaler.fit(train_df.values)
scaler.transform(test_array)  # OK
```

### 3. Convert Before Transform
```python
scaler.fit(train_df)
test_df = pd.DataFrame(test_array, columns=train_df.columns)
scaler.transform(test_df)  # OK
```

## Full Documentation
See [docs/STANDARDSCALER_FIX.md](docs/STANDARDSCALER_FIX.md) for:
- Detailed explanations
- 5 different solutions
- Best practices
- Common pitfalls
- Additional resources

## Run the Example
```bash
python src/text-summarizer/utils/preprocessing.py
```

## Run Tests
```bash
python -m unittest tests.test_preprocessing -v
```
