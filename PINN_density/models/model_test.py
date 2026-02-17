import numpy as np
import pandas as pd
import glob
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from transformers import TempInteractions, AdaptiveFeatureScaler, LearnableFeatureWeights
from config import TEST_CONFIG

class ModelTester:
    def __init__(self, config=TEST_CONFIG):
        self.config = config
        self.models = {}
        self.results = []
        
    def generate_synthetic_data(self):
        """Generate synthetic test data"""
        np.random.seed(self.config['random_seed'])
        
        # Temperature range (300K - 1200K)
        T_K = np.random.uniform(300, 1200, self.config['n_samples'])
        
        # Dopant types (categorical)
        dopant_types = ['Ca', 'Sr','Mg']
        dopant_A = np.random.choice(dopant_types, self.config['n_samples'])
        dopant_B = np.random.choice(dopant_types, self.config['n_samples'])
        
        # Molar fractions (0-0.5)
        xA_mol = np.random.uniform(0, 0.5, self.config['n_samples'])
        xB_mol = np.random.uniform(0, 0.5, self.config['n_samples'])
        xV_mol = np.random.uniform(0, 0.3, self.config['n_samples'])
        
        # Generate conductivity with physics-inspired relationships
        base_conductivity = (
            -2.0 + 
            0.005 * (T_K - 300) +  # Temperature activation
            2.0 * xA_mol +         # Primary dopant effect
            1.5 * xB_mol +         # Secondary dopant effect
            -3.0 * xV_mol          # Vacancy effect
        )
        
        # Add non-linear effects
        base_conductivity += 0.5 * np.sin(T_K/200)  # Temperature oscillation
        base_conductivity += 2.0 * xA_mol * xB_mol   # Dopant interaction
        
        # Add noise
        noise = np.random.normal(0, 0.2, self.config['n_samples'])
        log_conductivity = base_conductivity + noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'T_K': T_K,
            'dopant_A': dopant_A,
            'xA_mol': xA_mol,
            'dopant_B': dopant_B,
            'xB_mol': xB_mol,
            'xV_mol': xV_mol,
            'log(Conductivity(S/cm))': log_conductivity
        })
        
        print("\nSynthetic Data Statistics:")
        print("-------------------------")
        print(df.describe())
        print("\nSample of generated data:")
        print(df.head())
        return df
    
    def load_models(self):
        """Load models from joblib files"""
        print("Looking for .joblib files...")
        for model_path in glob.glob("*.joblib"):
            self._load_single_model(model_path)
            
    def _load_single_model(self, model_path):
        """Helper to load and validate single model"""
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        try:
            model = joblib.load(model_path)
            if hasattr(model, 'predict'):
                self.models[model_name] = model
                print(f"Successfully loaded model: {model_name}")
            else:
                self._try_reconstruct_pipeline(model_name, model)
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            
    def _build_pipeline_steps(self, step_list):
        """Helper to rebuild pipeline steps from serialized data"""
        steps = []
        for name, step_data in step_list:
            if isinstance(step_data, dict):
                if '__class__' in step_data:
                    # Handle scikit-learn estimators
                    class_name = step_data['__class__']
                    params = {k:v for k,v in step_data.items() if not k.startswith('__')}
                    
                    if class_name == 'TempInteractions':
                        step = TempInteractions(**params)
                    elif class_name == 'AdaptiveFeatureScaler':
                        step = AdaptiveFeatureScaler(**params)
                    elif class_name == 'LearnableFeatureWeights':
                        step = LearnableFeatureWeights(**params)
                    elif class_name == 'StandardScaler':
                        step = StandardScaler(**params)
                    else:
                        # Try to import and instantiate other sklearn estimators
                        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                        from sklearn.linear_model import Ridge, Lasso, ElasticNet
                        from sklearn.svm import SVR
                        from sklearn.neighbors import KNeighborsRegressor
                        from sklearn.tree import DecisionTreeRegressor
                        
                        estimator_map = {
                            'RandomForestRegressor': RandomForestRegressor,
                            'GradientBoostingRegressor': GradientBoostingRegressor,
                            'Ridge': Ridge,
                            'Lasso': Lasso,
                            'ElasticNet': ElasticNet,
                            'SVR': SVR,
                            'KNeighborsRegressor': KNeighborsRegressor,
                            'DecisionTreeRegressor': DecisionTreeRegressor
                        }
                        
                        if class_name in estimator_map:
                            step = estimator_map[class_name](**params)
                        else:
                            raise ValueError(f"Unknown estimator class: {class_name}")
                else:
                    # Handle dictionary-style transformers
                    if 'temp_col' in step_data:
                        step = TempInteractions(**step_data)
                    elif 'feature_weights' in step_data:
                        step = AdaptiveFeatureScaler(**step_data)
                    elif 'learning_rate' in step_data:
                        step = LearnableFeatureWeights(**step_data)
                    else:
                        step = step_data
            else:
                step = step_data
            steps.append((name, step))
        return steps

    def _try_reconstruct_pipeline(self, name, model_dict):
        """Try to reconstruct pipeline from dictionary"""
        try:
            if isinstance(model_dict, dict):
                if 'steps' in model_dict:
                    # Handle Pipeline
                    steps = self._build_pipeline_steps(model_dict['steps'])
                    pipeline = Pipeline(steps)
                    if hasattr(pipeline, 'predict'):
                        self.models[name] = pipeline
                        print(f"Successfully reconstructed pipeline for: {name}")
                        return
                elif '__class__' in model_dict:
                    # Handle single estimator
                    steps = self._build_pipeline_steps([('estimator', model_dict)])
                    if steps:
                        self.models[name] = steps[0][1]
                        print(f"Successfully reconstructed estimator for: {name}")
                        return
            print(f"Error: {name} is not a valid model or pipeline")
        except Exception as e:
            print(f"Pipeline reconstruction failed for {name}: {e}")
    
    def evaluate_models(self):
        """Evaluate all loaded models"""
        if not self.models:
            print("No valid models found!")
            return
            
        df = self.generate_synthetic_data()
        X = df.drop(self.config['target_column'], axis=1)
        y = df[self.config['target_column']]
        
        for name, model in self.models.items():
            self._evaluate_single_model(name, model, X, y)
            
        self._print_summary()
    
    def _evaluate_single_model(self, name, model, X, y):
        """Evaluate single model performance"""
        print(f"\nTesting {name}...")
        try:
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            self.results.append({
                'Model': name,
                'MAE': mae,
                'R2': r2
            })
            
            print(f"MAE: {mae:.4f}")
            print(f"R2:  {r2:.4f}")
            
            self._evaluate_feature_ranges(model, X, y)
            
        except Exception as e:
            print(f"Error testing {name}: {e}")
    
    def _evaluate_feature_ranges(self, model, X, y):
        """Evaluate model across feature ranges"""
        print("\nPerformance across feature ranges:")
        range_results = evaluate_model_ranges(model, X, y, self.config['feature_ranges'])
        for feature, metrics in range_results.items():
            avg_error = np.mean(metrics['errors'])
            print(f"{feature:20} Average MAE: {avg_error:.4f}")
            
    def _print_summary(self):
        """Print overall results summary"""
        if self.results:
            print("\nOverall Results Summary")
            print("=" * 50)
            results_df = pd.DataFrame(self.results)
            print(results_df.to_string(index=False))

def main():
    tester = ModelTester()
    print("\nStarting model testing...")
    print("=" * 50)
    
    tester.load_models()
    tester.evaluate_models()
    print("\nTesting completed!")

if __name__ == "__main__":
    main()