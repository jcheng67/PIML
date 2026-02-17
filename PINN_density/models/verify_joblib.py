import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import glob
import os

# Define TempInteractions class
class TempInteractions(BaseEstimator, TransformerMixin):
    """Temperature interaction features transformer"""
    def __init__(self, temp_col="T_K", num_cols=None):
        self.temp_col = temp_col
        self.num_cols = num_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num_cols_list = list(self.num_cols)
        n = len(X)

        # Temperature handling
        if isinstance(X, (pd.DataFrame,)) and (self.temp_col in X.columns):
            T = X[self.temp_col].astype(float).to_numpy().reshape(-1, 1)
        else:
            T = np.zeros((n, 1), dtype=float)

        invT = 1.0 / T
        logT = np.log(T)
        T2 = T**2
        temp_block = np.hstack([T, invT, logT, T2])

        # For numeric columns
        nums_cols_present = [c for c in num_cols_list if (isinstance(X, (pd.DataFrame,)) and c in X.columns)]
        if nums_cols_present:
            nums_present = X[nums_cols_present].astype(float).to_numpy()
            nums = np.zeros((n, len(num_cols_list)), dtype=float)
            for i, c in enumerate(num_cols_list):
                if c in nums_cols_present:
                    idx = nums_cols_present.index(c)
                    nums[:, i] = nums_present[:, idx]
        else:
            nums = np.zeros((n, len(num_cols_list)), dtype=float)

        inter = [t * nums for t in (T, invT, logT, T2)]
        inter_block = np.hstack(inter)
        return np.hstack([temp_block, nums, inter_block])

def verify_joblib(file_path, test_data):
    """
    Verify if a .joblib file is a valid model and can make predictions.
    
    Args:
        file_path (str): Path to the .joblib file.
        test_data (pd.DataFrame): Test data to use for predictions.
    
    Returns:
        None
    """
    try:
        # Load the .joblib file
        model_bundle = joblib.load(file_path)
        
        # Check if the file contains a valid model
        if hasattr(model_bundle, "predict"):
            print(f"Loaded model from {file_path}")
            
            # Test predictions
            predictions = model_bundle.predict(test_data)
            print(f"Predictions: {predictions[:5]}")  # Print first 5 predictions
        else:
            print(f"Error: {file_path} does not contain a valid model with a predict method.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.getcwd()
    print(f"Testing all .joblib files in the current directory: {current_dir}")
    
    # Create synthetic test data (replace with actual test data if available)
    test_data = pd.DataFrame({
        "T_K": np.random.uniform(300, 1200, 10),
        "dopant_A": ["Li"] * 10,
        "xA_mol": np.random.uniform(0, 0.5, 10),
        "dopant_B": ["Na"] * 10,
        "xB_mol": np.random.uniform(0, 0.5, 10),
        "xV_mol": np.random.uniform(0, 0.3, 10),
        "density_matrix (g/mL)": np.random.uniform(2, 6, 10),
    })
    
    # Iterate over all .joblib files in the current directory
    for joblib_file in glob.glob("*.joblib"):
        print(f"\nVerifying {joblib_file}...")
        verify_joblib(joblib_file, test_data)