"""
Tests for the preprocessing utilities.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
import warnings
import importlib.util

# Load the preprocessing module directly
try:
    # Try importing with hyphenated name first
    spec = importlib.util.spec_from_file_location(
        "preprocessing",
        os.path.join(os.path.dirname(__file__), '..', 'src', 'text-summarizer', 'utils', 'preprocessing.py')
    )
    preprocessing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocessing)
    
    FeatureScaler = preprocessing.FeatureScaler
    fix_standardscaler_warning = preprocessing.fix_standardscaler_warning
    fix_standardscaler_warning_alternative = preprocessing.fix_standardscaler_warning_alternative
    
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@unittest.skipUnless(SKLEARN_AVAILABLE, "sklearn not installed")
class TestFeatureScaler(unittest.TestCase):
    """Test cases for FeatureScaler class."""
    
    def setUp(self):
        """Set up test data."""
        self.train_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        self.test_array = np.array([[3, 30], [4, 40]])
        self.test_df = pd.DataFrame(self.test_array, columns=['feature1', 'feature2'])
    
    def test_fit_transform_with_dataframe(self):
        """Test fitting and transforming with DataFrame."""
        scaler = FeatureScaler()
        result = scaler.fit_transform(self.train_df)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.train_df.shape)
    
    def test_fit_with_dataframe_transform_with_array(self):
        """Test fitting with DataFrame and transforming with array (should not warn)."""
        scaler = FeatureScaler()
        scaler.fit(self.train_df)
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = scaler.transform(self.test_array)
            
            # Check no sklearn feature name warnings
            sklearn_warnings = [warning for warning in w 
                              if 'feature names' in str(warning.message).lower()]
            self.assertEqual(len(sklearn_warnings), 0, 
                           "Should not produce feature name warnings")
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_array.shape)
    
    def test_fit_with_array_transform_with_array(self):
        """Test fitting and transforming with arrays."""
        scaler = FeatureScaler()
        train_array = self.train_df.values
        scaler.fit(train_array)
        
        result = scaler.transform(self.test_array)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_array.shape)
    
    def test_inverse_transform(self):
        """Test inverse transform functionality."""
        scaler = FeatureScaler()
        scaler.fit(self.train_df)
        
        transformed = scaler.transform(self.test_array)
        inverse = scaler.inverse_transform(transformed)
        
        # Check values are close to original
        np.testing.assert_array_almost_equal(inverse, self.test_array, decimal=10)
    
    def test_mismatched_features_raises_error(self):
        """Test that mismatched features raise an error."""
        scaler = FeatureScaler()
        scaler.fit(self.train_df)
        
        # Try to transform with wrong number of features
        wrong_array = np.array([[1, 2, 3]])  # 3 features instead of 2
        
        with self.assertRaises(ValueError):
            scaler.transform(wrong_array)


@unittest.skipUnless(SKLEARN_AVAILABLE, "sklearn not installed")
class TestFixStandardScalerWarning(unittest.TestCase):
    """Test cases for fix_standardscaler_warning functions."""
    
    def setUp(self):
        """Set up test data."""
        self.train_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        self.test_array = np.array([[3, 30], [4, 40]])
        self.test_df = pd.DataFrame(self.test_array, columns=['feature1', 'feature2'])
    
    def test_fix_with_dataframe_and_array(self):
        """Test fixing with DataFrame training and array testing."""
        train_scaled, test_scaled = fix_standardscaler_warning(
            self.train_df, self.test_array
        )
        
        self.assertIsInstance(train_scaled, np.ndarray)
        self.assertIsInstance(test_scaled, np.ndarray)
        self.assertEqual(test_scaled.shape, self.test_array.shape)
    
    def test_fix_with_arrays(self):
        """Test fixing with both arrays."""
        train_array = self.train_df.values
        train_scaled, test_scaled = fix_standardscaler_warning(
            train_array, self.test_array
        )
        
        self.assertIsInstance(train_scaled, np.ndarray)
        self.assertIsInstance(test_scaled, np.ndarray)
    
    def test_fix_with_dataframes(self):
        """Test fixing with both DataFrames."""
        train_scaled, test_scaled = fix_standardscaler_warning(
            self.train_df, self.test_df
        )
        
        self.assertIsInstance(train_scaled, np.ndarray)
        self.assertIsInstance(test_scaled, np.ndarray)


@unittest.skipUnless(SKLEARN_AVAILABLE, "sklearn not installed")
class TestStandardScalerWarning(unittest.TestCase):
    """Test that the warning actually occurs without our fix."""
    
    def test_warning_occurs_without_fix(self):
        """Verify that the warning occurs when using StandardScaler incorrectly."""
        train_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30]
        })
        test_array = np.array([[2, 20]])
        
        scaler = StandardScaler()
        scaler.fit(train_df)
        
        # This should produce a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scaler.transform(test_array)
            
            # Check that we got a feature names warning
            feature_warnings = [warning for warning in w 
                              if 'feature names' in str(warning.message).lower()]
            self.assertGreater(len(feature_warnings), 0, 
                             "Should produce feature name warning")


if __name__ == '__main__':
    unittest.main()
