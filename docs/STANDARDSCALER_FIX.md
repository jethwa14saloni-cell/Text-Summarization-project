# StandardScaler Feature Names Warning - Solutions

## Problem

When using scikit-learn's `StandardScaler`, you may encounter the following warning:

```
UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
```

This warning occurs when:
1. You fit the StandardScaler with a pandas DataFrame (which has feature names/column names)
2. You then call transform with a numpy array (which doesn't have feature names)

## Why This Happens

Scikit-learn introduced feature name validation in version 1.0+ to help catch potential bugs where:
- The order of features might have changed between fit and transform
- Different features are being used between training and testing
- Column names don't match, indicating a potential data pipeline issue

## Solutions

### Solution 1: Use the FeatureScaler Wrapper Class (Recommended)

The repository includes a `FeatureScaler` wrapper class that automatically handles feature name consistency:

```python
from text-summarizer.utils.preprocessing import FeatureScaler
import pandas as pd
import numpy as np

# Create sample data
train_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
test_array = np.array([[1.5, 4.5]])

# Use FeatureScaler - no warning!
scaler = FeatureScaler()
scaler.fit(train_df)
scaled_test = scaler.transform(test_array)  # Automatically converts to DataFrame internally
```

### Solution 2: Keep Data Types Consistent

Ensure you use the same data type (DataFrame or array) for both fit and transform:

#### Option A: Use DataFrames for Both
```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Training data as DataFrame
train_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

# Test data also as DataFrame with same column names
test_df = pd.DataFrame({'feature1': [1.5], 'feature2': [4.5]})

scaler = StandardScaler()
scaler.fit(train_df)
scaled_test = scaler.transform(test_df)  # No warning!
```

#### Option B: Use Arrays for Both
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Training data as array
train_array = np.array([[1, 4], [2, 5], [3, 6]])

# Test data also as array
test_array = np.array([[1.5, 4.5]])

scaler = StandardScaler()
scaler.fit(train_array)
scaled_test = scaler.transform(test_array)  # No warning!
```

### Solution 3: Convert Arrays to DataFrames Before Transform

If you must use arrays but fitted with a DataFrame, convert the array to a DataFrame:

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Fit with DataFrame
train_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
scaler = StandardScaler()
scaler.fit(train_df)

# Convert array to DataFrame before transform
test_array = np.array([[1.5, 4.5]])
test_df = pd.DataFrame(test_array, columns=train_df.columns)
scaled_test = scaler.transform(test_df)  # No warning!
```

### Solution 4: Use set_output (sklearn >= 1.2)

For sklearn version 1.2 and above, you can use `set_output`:

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
test_df = pd.DataFrame({'feature1': [1.5], 'feature2': [4.5]})

# Configure scaler to output pandas DataFrames
scaler = StandardScaler().set_output(transform="pandas")
scaler.fit(train_df)
scaled_test = scaler.transform(test_df)  # Returns DataFrame, no warning!
```

### Solution 5: Convert DataFrame to Array Before Fit

If you don't need feature names, convert to arrays from the start:

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
test_df = pd.DataFrame({'feature1': [1.5], 'feature2': [4.5]})

scaler = StandardScaler()
scaler.fit(train_df.values)  # Use .values to get numpy array
scaled_test = scaler.transform(test_df.values)  # No warning!
```

## Testing Your Fix

Run the example script to see the warning and solutions in action:

```bash
python src/text-summarizer/utils/preprocessing.py
```

## Best Practices

1. **Be Consistent**: Use the same data type throughout your pipeline
2. **Use DataFrames When Possible**: They provide better debugging with feature names
3. **Document Feature Order**: If using arrays, document the expected feature order
4. **Use the FeatureScaler Wrapper**: It handles conversions automatically and safely
5. **Validate Feature Names**: Always ensure column names match between train and test

## Common Pitfalls to Avoid

❌ **Don't Do This:**
```python
scaler.fit(train_df)  # DataFrame
scaler.transform(test_array)  # Array - causes warning!
```

✅ **Do This Instead:**
```python
scaler.fit(train_df)  # DataFrame
test_df = pd.DataFrame(test_array, columns=train_df.columns)
scaler.transform(test_df)  # DataFrame - no warning!
```

## Additional Resources

- [Scikit-learn Feature Names Documentation](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html)
- [StandardScaler API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
