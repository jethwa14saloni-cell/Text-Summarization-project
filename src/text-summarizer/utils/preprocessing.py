"""
Preprocessing utilities for text summarization project.
This module demonstrates the proper way to handle sklearn StandardScaler
to avoid feature name warnings.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


class FeatureScaler:
    """
    A wrapper class for sklearn StandardScaler that properly handles feature names
    to avoid the warning: "X does not have valid feature names, but StandardScaler 
    was fitted with feature names"
    
    The warning occurs when:
    1. StandardScaler is fitted with a pandas DataFrame (which has feature names)
    2. Transform is called with a numpy array (which doesn't have feature names)
    
    This class ensures consistent data types between fit and transform operations.
    """
    
    def __init__(self):
        """Initialize the FeatureScaler with a StandardScaler instance."""
        try:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self._feature_names = None
            self._fitted_with_dataframe = False
        except ImportError:
            raise ImportError(
                "sklearn is required for FeatureScaler. "
                "Install it with: pip install scikit-learn"
            )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'FeatureScaler':
        """
        Fit the scaler to the data.
        
        Args:
            X: Training data (pandas DataFrame or numpy array)
            
        Returns:
            self: Fitted scaler instance
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            self._fitted_with_dataframe = True
            self.scaler.fit(X)
        else:
            self._fitted_with_dataframe = False
            self.scaler.fit(X)
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform the data using the fitted scaler.
        
        This method handles the conversion to ensure consistent feature names
        between fit and transform operations, preventing sklearn warnings.
        
        Args:
            X: Data to transform (pandas DataFrame or numpy array)
            
        Returns:
            Transformed data as numpy array
        """
        # If fitted with DataFrame, ensure transform also uses DataFrame
        if self._fitted_with_dataframe:
            if isinstance(X, np.ndarray):
                # Convert numpy array to DataFrame with stored feature names
                if self._feature_names is not None and X.shape[1] == len(self._feature_names):
                    X = pd.DataFrame(X, columns=self._feature_names)
                else:
                    raise ValueError(
                        f"Cannot transform: Expected {len(self._feature_names) if self._feature_names else 'unknown'} "
                        f"features, got {X.shape[1] if X.ndim > 1 else 1}"
                    )
            elif isinstance(X, pd.DataFrame):
                # Ensure column names match
                if not all(col in self._feature_names for col in X.columns):
                    raise ValueError(
                        f"Column names mismatch. Expected: {self._feature_names}, "
                        f"Got: {X.columns.tolist()}"
                    )
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit the scaler and transform the data in one step.
        
        Args:
            X: Training data (pandas DataFrame or numpy array)
            
        Returns:
            Transformed data as numpy array
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Inverse transform the scaled data back to original scale.
        
        Args:
            X: Scaled data to inverse transform
            
        Returns:
            Data in original scale as numpy array
        """
        # Handle the same way as transform for consistency
        if self._fitted_with_dataframe and isinstance(X, np.ndarray):
            if self._feature_names is not None and X.shape[1] == len(self._feature_names):
                X = pd.DataFrame(X, columns=self._feature_names)
        
        return self.scaler.inverse_transform(X)


def fix_standardscaler_warning(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray]
) -> tuple:
    """
    Utility function demonstrating how to properly use StandardScaler
    to avoid feature name warnings.
    
    Solution 1: Ensure consistent data types between fit and transform
    
    Args:
        X_train: Training data
        X_test: Test data
        
    Returns:
        Tuple of (scaled_train, scaled_test)
    """
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "sklearn is required. Install it with: pip install scikit-learn"
        )
    
    scaler = StandardScaler()
    
    # Method 1: If training with DataFrame, ensure test is also DataFrame
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, np.ndarray):
        # Convert test array to DataFrame with same columns
        X_test = pd.DataFrame(X_test, columns=X_train.columns)
    
    # Method 2: If training with array, ensure test is also array
    elif isinstance(X_train, np.ndarray) and isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    # Now fit and transform with consistent types
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def fix_standardscaler_warning_alternative(
    X_train: pd.DataFrame,
    X_test: Union[pd.DataFrame, np.ndarray]
) -> tuple:
    """
    Alternative solution: Use set_output to configure sklearn to work with pandas.
    This requires sklearn >= 1.2
    
    Args:
        X_train: Training data as DataFrame
        X_test: Test data
        
    Returns:
        Tuple of (scaled_train, scaled_test)
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn import __version__
        
        # Check if set_output is available (sklearn >= 1.2)
        major, minor = map(int, __version__.split('.')[:2])
        if major > 1 or (major == 1 and minor >= 2):
            scaler = StandardScaler().set_output(transform="pandas")
            
            # Convert test to DataFrame if needed
            if isinstance(X_test, np.ndarray):
                X_test = pd.DataFrame(X_test, columns=X_train.columns)
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled
        else:
            # Fall back to regular fix
            return fix_standardscaler_warning(X_train, X_test)
            
    except ImportError:
        raise ImportError(
            "sklearn is required. Install it with: pip install scikit-learn"
        )


# Example usage function
def example_usage():
    """
    Example demonstrating how to avoid StandardScaler warnings.
    """
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("sklearn not installed. Install with: pip install scikit-learn")
        return
    
    # Create sample data
    train_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    
    # PROBLEM: This causes the warning
    print("=" * 60)
    print("PROBLEM: Fitting with DataFrame, transforming with numpy array")
    print("=" * 60)
    scaler_problem = StandardScaler()
    scaler_problem.fit(train_data)
    
    # This will cause warning
    test_array = np.array([[3, 30], [4, 40]])
    print("Transforming numpy array (causes warning)...")
    try:
        scaled_test = scaler_problem.transform(test_array)
        print("Result shape:", scaled_test.shape)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("SOLUTION 1: Using FeatureScaler wrapper class")
    print("=" * 60)
    scaler_fixed = FeatureScaler()
    scaler_fixed.fit(train_data)
    
    # Now this works without warning
    print("Transforming numpy array (no warning)...")
    scaled_test = scaler_fixed.transform(test_array)
    print("Result shape:", scaled_test.shape)
    
    print("\n" + "=" * 60)
    print("SOLUTION 2: Keep consistent data types")
    print("=" * 60)
    scaler_consistent = StandardScaler()
    
    # Option A: Both as DataFrames
    test_df = pd.DataFrame(test_array, columns=train_data.columns)
    scaler_consistent.fit(train_data)
    scaled_test = scaler_consistent.transform(test_df)
    print("Transforming DataFrame (no warning)...")
    print("Result shape:", scaled_test.shape)
    
    # Option B: Both as arrays
    print("\nOr use arrays for both:")
    scaler_array = StandardScaler()
    scaler_array.fit(train_data.values)
    scaled_test = scaler_array.transform(test_array)
    print("Result shape:", scaled_test.shape)


if __name__ == "__main__":
    example_usage()
