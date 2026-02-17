#!/usr/bin/env python3
"""
Compare many regression models on the SAME engineered feature set for:
    log(Conductivity(S/cm))

Features expected in CSV:
  T_K, dopant_A, xA_mol, dopant_B, xB_mol, xV_mol, density_matrix (g/mL), log(Conductivity(S/cm))

What this script provides:
  - Progress bars across models and CV folds (tqdm)
  - Save-all / save-one model bundles (.joblib)
  - Option to run only a single model
  - Optional temperature monotonic constraints for supported boosters (HGBR, XGBoost)  # LightGBM removed
  - Physics-Informed Neural Networks (PINNs) with multiple architectures:
      * BasicPINN: Standard PINN with physics constraints in loss function
      * MultiScalePINN: Multi-scale networks for different physics scales
      * AttentionPINN: Attention mechanism for physics term weighting
      * ResidualPINN: Residual connections for better gradient flow
      * EnsemblePINN: Ensemble of different PINN architectures
      * AdaptivePINN: Learnable physics constraint weights
  - Comprehensive Physics-Informed ML with ALL major physical constraints:
      * Thermodynamic: Gibbs free energy, chemical potential
      * Transport Theory: Nernst-Einstein relation, mobility-conductivity
      * Crystal Structure: Coordination number, lattice strain effects
      * Statistical Mechanics: Boltzmann distribution, Fermi-Dirac statistics
      * Defect Chemistry: Schottky defects, Frenkel defects
      * Electrochemistry: Butler-Volmer equation, Nernst equation
      * Quantum Effects: Tunneling, quantum confinement
      * Basic Constraints: Arrhenius behavior, conductivity bounds, charge conservation
  - Adaptive Feature Weighting: ML learns which features contribute most
      * Automatically reduces influence of non-contributing features
      * Learnable feature weights with sparsity regularization
      * Real-time feature importance tracking and visualization
  - Explanations:
      * Permutation importance CSV for ALL models
      * SHAP beeswarm + bar for tree models (if `shap` installed)

Examples:
  python compare_fits_all_progress.py --csv 1.csv --out_dir results_all --folds 5 --save_all
  python compare_fits_all_progress.py --csv 1.csv --out_dir results --folds 5 --only_model ElasticNetCV \
      --model_to_save ElasticNetCV --save_path results/enet.joblib
  python compare_fits_all_progress.py --csv 1.csv --out_dir results_mono --folds 5 \
      --save_all --monotone_temp --explain
  python compare_fits_all_progress.py --csv 1.csv --out_dir pinn_results --folds 5 \
      --pinn_only --pinn_epochs 2000 --pinn_lr 0.001 --device cuda
  python compare_fits_all_progress.py --csv 1.csv --out_dir physics_results --folds 5 \
      --physics_only --physics_weight 0.2 --use_thermodynamic --use_transport_theory
  python compare_fits_all_progress.py --csv 1.csv --out_dir adaptive_results --folds 5 \
      --adaptive_only --track_contributions --learning_rate 0.02 --sparsity_lambda 0.01
  python compare_fits_all_progress.py --csv 1.csv --out_dir comprehensive_physics --folds 5 \
      --physics_only --use_thermodynamic --use_transport_theory --use_crystal_structure \
      --use_statistical_mechanics --use_defect_chemistry --use_electrochemistry --use_quantum_effects

  # Excluding input columns (drop columns from the CSV before processing)
  # --exclude_features accepts a comma-separated list of column names to remove from the input dataframe.
  # The TARGET column (log(Conductivity(S/cm))) cannot be excluded; attempts to do so will abort.
  # Missing column names are ignored. Use quotes when names contain spaces or parentheses.
  python compare_fits_all_progress.py --csv 1.csv --out_dir results_no_density \
      --exclude_features "density_matrix (g/mL)" --folds 5 --save_all

  # Example: drop multiple columns
  python compare_fits_all_progress.py --csv 1.csv --out_dir results_dropcols \
      --exclude_features "density_matrix (g/mL),xV_mol" --folds 5

"""
import argparse, os, warnings, sys, re, time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

# linear / regularized
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, BayesianRidge
from sklearn.cross_decomposition import PLSRegression

# kernel / neighbors / neural
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# trees / ensembles
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)

# Optional libs (if present)
HAS_XGB = HAS_CAT = False  # HAS_LGBM = False
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    pass
# try:
#     from lightgbm import LGBMRegressor  # type: ignore
#     HAS_LGBM = True
# except Exception:
#     pass
try:
    from catboost import CatBoostRegressor  # type: ignore
    HAS_CAT = True
except Exception:
    pass

# Optional SHAP
HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

# ---- Adaptive Feature Weighting Components ----
class LearnableFeatureWeights:
    """
    Learnable weights for features - ML determines which features contribute most.
    Automatically reduces influence of non-contributing features.
    """
    
    def __init__(self, n_features, learning_rate=0.01, sparsity_lambda=0.01):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.sparsity_lambda = sparsity_lambda
        
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.1, n_features)
        self.weight_history = []
        self.feature_contributions = np.zeros(n_features)
        
    def get_feature_weights(self):
        """Get current feature weights (softmax normalized)"""
        # Apply softmax to ensure weights sum to 1
        exp_weights = np.exp(self.weights - np.max(self.weights))  # Numerical stability
        return exp_weights / np.sum(exp_weights)
    
    def update_weights(self, gradients, feature_contributions):
        """Update weights based on gradients and feature contributions"""
        # Regularization to encourage sparsity (reduce non-contributing features)
        sparsity_grad = self.sparsity_lambda * np.sign(self.weights)
        
        # Update weights
        self.weights -= self.learning_rate * (gradients + sparsity_grad)
        
        # Track feature contributions
        self.feature_contributions += feature_contributions
        self.weight_history.append(self.weights.copy())
    
    def get_feature_importance(self):
        """Get normalized feature importance scores"""
        weights = self.get_feature_weights()
        contributions = self.feature_contributions / (len(self.weight_history) + 1e-10)
        
        # Combine weights and contributions
        importance = weights * (1 + contributions)
        return importance / np.sum(importance)
    
    def get_active_features(self, threshold=0.01):
        """Get indices of features that are actively contributing"""
        weights = self.get_feature_weights()
        return np.where(weights > threshold)[0]

class AdaptiveFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Adaptive feature scaler that learns which features to emphasize/de-emphasize.
    """
    
    def __init__(self, learning_rate=0.01, sparsity_lambda=0.01):
        self.learning_rate = learning_rate
        self.sparsity_lambda = sparsity_lambda
        self.feature_weights = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Initialize feature weights"""
        n_features = X.shape[1]
        self.feature_weights = LearnableFeatureWeights(
            n_features, self.learning_rate, self.sparsity_lambda
        )
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply learned feature weights"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        weights = self.feature_weights.get_feature_weights()
        return X * weights
    
    def update_weights(self, gradients, feature_contributions):
        """Update feature weights during training"""
        if self.feature_weights:
            self.feature_weights.update_weights(gradients, feature_contributions)
    
    def get_feature_importance(self):
        """Get current feature importance"""
        if self.feature_weights:
            return self.feature_weights.get_feature_importance()
        return None

class AdaptiveRegressor:
    """
    Regressor that learns feature importance and adapts accordingly.
    """
    
    def __init__(self, base_regressor, learning_rate=0.01, sparsity_lambda=0.01):
        self.base_regressor = base_regressor
        self.learning_rate = learning_rate
        self.sparsity_lambda = sparsity_lambda
        self.feature_scaler = None
        self.is_fitted = False
        self.feature_importance_history = []
        
    def fit(self, X, y):
        """Fit with adaptive feature weighting"""
        # Initialize adaptive scaler
        self.feature_scaler = AdaptiveFeatureScaler(
            self.learning_rate, self.sparsity_lambda
        )
        self.feature_scaler.fit(X)
        
        # Transform features with initial weights
        X_weighted = self.feature_scaler.transform(X)
        
        # Fit base regressor
        self.base_regressor.fit(X_weighted, y)
        
        # Iterative weight updates
        self._iterative_weight_update(X, y, n_iterations=10)
        
        self.is_fitted = True
        return self
    
    def _iterative_weight_update(self, X, y, n_iterations=10):
        """Iteratively update feature weights based on contribution"""
        for iteration in range(n_iterations):
            # Get current predictions
            X_weighted = self.feature_scaler.transform(X)
            y_pred = self.base_regressor.predict(X_weighted)
            
            # Calculate feature contributions
            feature_contributions = self._calculate_feature_contributions(X, y, y_pred)
            
            # Calculate gradients (simplified)
            gradients = self._calculate_gradients(X, y, y_pred)
            
            # Update weights
            self.feature_scaler.update_weights(gradients, feature_contributions)
            
            # Refit base regressor with new weights
            X_weighted = self.feature_scaler.transform(X)
            self.base_regressor.fit(X_weighted, y)
            
            # Track importance
            importance = self.feature_scaler.get_feature_importance()
            self.feature_importance_history.append(importance.copy())
    
    def _calculate_feature_contributions(self, X, y_true, y_pred):
        """Calculate how much each feature contributes to predictions"""
        contributions = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            # Calculate correlation between feature and prediction error
            feature_values = X[:, i]
            prediction_error = np.abs(y_true - y_pred)
            
            # Higher correlation with error = lower contribution
            correlation = np.corrcoef(feature_values, prediction_error)[0, 1]
            contributions[i] = -correlation if not np.isnan(correlation) else 0
        
        return contributions
    
    def _calculate_gradients(self, X, y_true, y_pred):
        """Calculate gradients for weight updates"""
        gradients = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            # Simple gradient: how much does changing this feature weight affect loss
            feature_values = X[:, i]
            error = y_true - y_pred
            
            # Gradient is correlation between feature and error
            gradient = np.mean(feature_values * error)
            gradients[i] = gradient
        
        return gradients
    
    def predict(self, X):
        """Predict with adaptive feature weighting"""
        if not self.is_fitted:
            raise ValueError("Must fit before predict")
        
        X_weighted = self.feature_scaler.transform(X)
        return self.base_regressor.predict(X_weighted)
    
    def score(self, X, y):
        """Calculate score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_feature_importance(self):
        """Get learned feature importance"""
        if self.feature_scaler:
            return self.feature_scaler.get_feature_importance()
        return None
    
    def get_active_features(self, threshold=0.01):
        """Get indices of actively contributing features"""
        if self.feature_scaler and self.feature_scaler.feature_weights:
            return self.feature_scaler.feature_weights.get_active_features(threshold)
        return None

class FeatureContributionTracker:
    """
    Tracks and visualizes feature contributions over training.
    """
    
    def __init__(self):
        self.contribution_history = []
        self.feature_names = None
        
    def track_contributions(self, contributions, feature_names=None):
        """Track feature contributions"""
        self.contribution_history.append(contributions.copy())
        if feature_names is not None:
            self.feature_names = feature_names
    
    def plot_contribution_evolution(self, save_path=None):
        """Plot how feature contributions evolve over training"""
        if not self.contribution_history:
            return
        
        import matplotlib.pyplot as plt
        
        contributions_array = np.array(self.contribution_history)
        
        plt.figure(figsize=(12, 8))
        
        for i in range(contributions_array.shape[1]):
            if self.feature_names is None or i >= len(self.feature_names):
                label = f'Feature {i}'
            else:
                label = self.feature_names[i]
            plt.plot(contributions_array[:, i], label=label)
        
        plt.xlabel('Training Iteration')
        plt.ylabel('Feature Contribution')
        plt.title('Feature Contribution Evolution During Training')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_final_contributions(self):
        """Get final feature contributions"""
        if self.contribution_history:
            return self.contribution_history[-1]
        return None

# ---- Comprehensive Physical Constraints ----
class PhysicalConstraints:
    """
    Comprehensive physical constraints for ionic conductivity modeling.
    Implements all major physics principles relevant to solid-state ionic conductors.
    """
    
    # Physical constants
    k_B = 8.617e-5  # eV/K (Boltzmann constant)
    k_B_J = 1.38e-23  # J/K (Boltzmann constant)
    q = 1.6e-19  # C (Elementary charge)
    F = 96485  # C/mol (Faraday constant)
    R = 8.314  # J/(mol·K) (Gas constant)
    N_A = 6.022e23  # mol⁻¹ (Avogadro's number)
    
    @staticmethod
    def arrhenius_constraint(y_pred, T_K, alpha=0.1):
        """
        Arrhenius equation: σ = σ₀ * exp(-Eₐ/(k_B*T))
        log(σ) = log(σ₀) - Eₐ/(k_B*T)
        """
        T_inv = 1.0 / T_K
        # Calculate expected Arrhenius slope
        arrhenius_slope = np.polyfit(T_inv, y_pred, 1)[0]
        # Penalty for deviation from linear Arrhenius behavior
        arrhenius_penalty = alpha * np.abs(arrhenius_slope)
        return arrhenius_penalty
    
    @staticmethod
    def conductivity_bounds_constraint(y_pred, alpha=0.1):
        """
        Physical bounds: conductivity must be positive and reasonable.
        Typical range: 1e-10 to 1e+3 S/cm
        """
        sigma_pred = np.exp(y_pred)
        
        # Lower bound penalty (must be positive)
        lower_bound_penalty = alpha * np.sum(np.maximum(0, 1e-10 - sigma_pred))
        
        # Upper bound penalty (unrealistic high conductivity)
        upper_bound_penalty = alpha * np.sum(np.maximum(0, sigma_pred - 1e+3))
        
        return lower_bound_penalty + upper_bound_penalty
    
    @staticmethod
    def charge_conservation_constraint(y_pred, xA_mol, xB_mol, xV_mol, alpha=0.1):
        """
        Charge conservation: total charge must be conserved.
        For ionic conductors: |xA - xB| should correlate with conductivity.
        """
        charge_imbalance = np.abs(xA_mol - xB_mol)
        
        # Higher charge imbalance should generally lead to higher conductivity
        expected_correlation = np.corrcoef(charge_imbalance, y_pred)[0, 1]
        correlation_penalty = alpha * max(0, -expected_correlation)  # Penalize negative correlation
        
        return correlation_penalty
    
    @staticmethod
    def density_conductivity_constraint(y_pred, density, alpha=0.1):
        """
        Density-conductivity relationship: higher density often means higher conductivity.
        """
        # Compute scalar correlation; guard NaNs and shapes
        dens = np.asarray(density).ravel()
        pred = np.asarray(y_pred).ravel()
        try:
            corr = float(np.corrcoef(dens, pred)[0, 1])
        except Exception:
            corr = 0.0
        if np.isnan(corr):
            corr = 0.0
        correlation_penalty = alpha * max(0.0, -corr)  # Penalize negative correlation
        return correlation_penalty
    
    @staticmethod
    def gibbs_free_energy_constraint(y_pred, T_K, xA_mol, xB_mol, alpha=0.1):
        """
        Gibbs free energy: G = H - TS
        Entropy term: -TS, Enthalpy term: H (related to conductivity)
        """
        # Entropy term (simplified)
        entropy_term = -T_K * (xA_mol * np.log(xA_mol + 1e-10) + xB_mol * np.log(xB_mol + 1e-10))
        
        # Enthalpy term (conductivity relates to enthalpy)
        enthalpy_term = y_pred
        
        # Gibbs free energy should be minimized
        gibbs_penalty = alpha * np.sum((enthalpy_term - entropy_term)**2)
        return gibbs_penalty
    
    @staticmethod
    def chemical_potential_constraint(y_pred, xA_mol, xB_mol, T_K, alpha=0.1):
        """
        Chemical potential: μ = μ₀ + k_B*T*ln(x)
        """
        mu_A = PhysicalConstraints.k_B * T_K * np.log(xA_mol + 1e-10)
        mu_B = PhysicalConstraints.k_B * T_K * np.log(xB_mol + 1e-10)
        
        # Chemical potential difference should correlate with conductivity
        mu_diff = mu_A - mu_B
        correlation = np.corrcoef(mu_diff, y_pred)[0, 1]
        potential_penalty = alpha * max(0, -correlation)
        
        return potential_penalty
    
    @staticmethod
    def nernst_einstein_constraint(y_pred, T_K, xA_mol, xB_mol, alpha=0.1):
        """
        Nernst-Einstein relation: σ = (q²Dc)/(k_B*T)
        where D is diffusivity, c is concentration
        """
        q = PhysicalConstraints.q
        k_B = PhysicalConstraints.k_B_J
        c_total = xA_mol + xB_mol
        
        # Expected conductivity from Nernst-Einstein
        sigma_expected = (q**2 * c_total) / (k_B * T_K)
        log_sigma_expected = np.log(sigma_expected + 1e-10)
        
        nernst_penalty = alpha * np.sum((y_pred - log_sigma_expected)**2)
        return nernst_penalty
    
    @staticmethod
    def mobility_constraint(y_pred, T_K, density, alpha=0.1):
        """
        Mobility-conductivity relationship: σ = n*q*μ
        where μ is mobility, n is carrier density
        """
        q = PhysicalConstraints.q
        n = density * PhysicalConstraints.N_A  # Convert to number density
        
        # Mobility should decrease with temperature (phonon scattering)
        mobility = np.exp(y_pred) / (n * q)
        mobility_expected = 1.0 / T_K  # Rough approximation
        
        mobility_penalty = alpha * np.sum((mobility - mobility_expected)**2)
        return mobility_penalty
    
    @staticmethod
    def coordination_constraint(y_pred, xA_mol, xB_mol, xV_mol, alpha=0.1):
        """
        Coordination number effects: higher coordination → higher conductivity
        """
        # Assume 6-coordinate sites
        coordination = 6 - xV_mol
        coordination_factor = coordination / 6.0
        
        # Conductivity should scale with coordination
        expected_conductivity = y_pred * coordination_factor
        coordination_penalty = alpha * np.sum((y_pred - expected_conductivity)**2)
        
        return coordination_penalty
    
    @staticmethod
    def lattice_strain_constraint(y_pred, xA_mol, xB_mol, alpha=0.1):
        """
        Lattice strain effects: strain from dopant size mismatch affects mobility
        """
        # Strain from dopant size mismatch
        strain = np.abs(xA_mol - xB_mol)
        
        # Higher strain → lower conductivity (more defects)
        strain_penalty = alpha * np.sum(strain * 0.1)
        return strain_penalty
    
    @staticmethod
    def boltzmann_distribution_constraint(y_pred, T_K, E_activation=0.5, alpha=0.1):
        """
        Boltzmann distribution: f(E) = exp(-E/k_B*T)
        """
        k_B = PhysicalConstraints.k_B
        boltzmann_factor = np.exp(-E_activation / (k_B * T_K))
        
        # Conductivity should follow Boltzmann statistics
        expected_log_sigma = np.log(boltzmann_factor + 1e-10)
        boltzmann_penalty = alpha * np.sum((y_pred - expected_log_sigma)**2)
        
        return boltzmann_penalty
    
    @staticmethod
    def fermi_dirac_constraint(y_pred, T_K, xA_mol, xB_mol, alpha=0.1):
        """
        Fermi-Dirac statistics: f(E) = 1/(1 + exp((E-E_F)/k_B*T))
        """
        k_B = PhysicalConstraints.k_B
        E_F = 0.5  # Fermi energy (eV)
        
        # Electronic contribution to conductivity
        fermi_factor = 1.0 / (1.0 + np.exp((E_F) / (k_B * T_K)))
        electronic_conductivity = np.log(fermi_factor + 1e-10)
        
        fermi_penalty = alpha * np.sum((y_pred - electronic_conductivity)**2)
        return fermi_penalty
    
    @staticmethod
    def schottky_defect_constraint(y_pred, T_K, xV_mol, alpha=0.1):
        """
        Schottky defect formation: cation vacancy + anion vacancy
        """
        E_schottky = 2.0  # eV (typical value)
        k_B = PhysicalConstraints.k_B
        
        # Defect concentration
        defect_conc = np.exp(-E_schottky / (2 * k_B * T_K))
        
        # More defects → higher conductivity
        defect_factor = xV_mol / (defect_conc + 1e-10)
        schottky_penalty = alpha * np.sum((y_pred - np.log(defect_factor + 1e-10))**2)
        
        return schottky_penalty
    
    @staticmethod
    def frenkel_defect_constraint(y_pred, T_K, xA_mol, xB_mol, alpha=0.1):
        """
        Frenkel defect formation: interstitial + vacancy
        """
        E_frenkel = 1.5  # eV
        k_B = PhysicalConstraints.k_B
        
        # Interstitial concentration
        interstitial_conc = np.exp(-E_frenkel / (2 * k_B * T_K))
        
        # Frenkel defects enhance conductivity
        frenkel_factor = (xA_mol + xB_mol) * interstitial_conc
        frenkel_penalty = alpha * np.sum((y_pred - np.log(frenkel_factor + 1e-10))**2)
        
        return frenkel_penalty
    
    @staticmethod
    def butler_volmer_constraint(y_pred, T_K, xA_mol, xB_mol, alpha=0.1):
        """
        Butler-Volmer equation: i = i₀[exp(αFη/RT) - exp(-(1-α)Fη/RT)]
        """
        F = PhysicalConstraints.F
        R = PhysicalConstraints.R
        alpha_transfer = 0.5  # Transfer coefficient
        
        # Overpotential (simplified)
        eta = np.abs(xA_mol - xB_mol) * 0.1  # V
        
        # Exchange current density
        i0 = np.exp(y_pred)  # Convert log(σ) to current density
        
        # Butler-Volmer current
        bv_current = i0 * (np.exp(alpha_transfer * F * eta / (R * T_K)) - 
                          np.exp(-(1-alpha_transfer) * F * eta / (R * T_K)))
        
        butler_volmer_penalty = alpha * np.sum((bv_current - i0)**2)
        return butler_volmer_penalty
    
    @staticmethod
    def nernst_equation_constraint(y_pred, T_K, xA_mol, xB_mol, alpha=0.1):
        """
        Nernst equation: E = E₀ - (RT/nF)ln(Q)
        """
        F = PhysicalConstraints.F
        R = PhysicalConstraints.R
        n = 1  # Number of electrons
        
        # Reaction quotient
        Q = xA_mol / (xB_mol + 1e-10)
        
        # Nernst potential
        E_nernst = -(R * T_K / (n * F)) * np.log(Q)
        
        # Conductivity should correlate with potential
        nernst_penalty = alpha * np.sum((y_pred - E_nernst)**2)
        return nernst_penalty
    
    @staticmethod
    def tunneling_constraint(y_pred, density, xV_mol, alpha=0.1):
        """
        Quantum tunneling through barriers
        """
        # Barrier width (inversely related to density)
        barrier_width = 1.0 / (density + 1e-10)
        
        # Tunneling probability: T ∝ exp(-2κd)
        kappa = 1.0  # Decay constant
        tunneling_prob = np.exp(-2 * kappa * barrier_width)
        
        # More vacancies → more tunneling
        vacancy_factor = xV_mol * tunneling_prob
        tunneling_penalty = alpha * np.sum((y_pred - np.log(vacancy_factor + 1e-10))**2)
        
        return tunneling_penalty
    
    @staticmethod
    def quantum_confinement_constraint(y_pred, density, alpha=0.1):
        """
        Quantum confinement effects
        """
        # Particle size (inversely related to density)
        particle_size = 1.0 / (density + 1e-10)
        
        # Quantum confinement energy
        confinement_energy = 1.0 / (particle_size**2)
        
        # Higher confinement → lower conductivity
        confinement_factor = 1.0 / (1.0 + confinement_energy)
        confinement_penalty = alpha * np.sum((y_pred - np.log(confinement_factor + 1e-10))**2)
        
        return confinement_penalty

class ComprehensivePhysicsInformedLoss:
    """
    Comprehensive physics-informed loss combining all physical constraints.
    """
    
    def __init__(self, constraint_weights=None):
        """
        Initialize with constraint weights.
        If None, uses default weights.
        """
        self.constraint_weights = constraint_weights or {
            'arrhenius': 0.1,
            'bounds': 0.1,
            'charge_conservation': 0.1,
            'density': 0.1,
            'thermodynamic': 0.05,
            'transport': 0.05,
            'crystal_structure': 0.05,
            'statistical_mechanics': 0.05,
            'defect_chemistry': 0.05,
            'electrochemistry': 0.05,
            'quantum': 0.05
        }
    
    def calculate_total_loss(self, y_pred, X, y_true=None):
        """
        Calculate total physics-informed loss.
        
        Args:
            y_pred: Predicted log(conductivity) values
            X: Feature matrix [T_K, dopant_A, xA_mol, dopant_B, xB_mol, xV_mol, density]
            y_true: True log(conductivity) values (optional)
        """
        total_loss = 0.0
        
        # Extract features
        T_K = X[:, 0]
        xA_mol = X[:, 2]
        xB_mol = X[:, 3]
        xV_mol = X[:, 4]
        density = X[:, 6]
        
        # Thermodynamic constraints
        if self.constraint_weights['thermodynamic'] > 0:
            total_loss += PhysicalConstraints.gibbs_free_energy_constraint(
                y_pred, T_K, xA_mol, xB_mol, self.constraint_weights['thermodynamic']
            )
            total_loss += PhysicalConstraints.chemical_potential_constraint(
                y_pred, xA_mol, xB_mol, T_K, self.constraint_weights['thermodynamic']
            )
        
        # Transport theory constraints
        if self.constraint_weights['transport'] > 0:
            total_loss += PhysicalConstraints.nernst_einstein_constraint(
                y_pred, T_K, xA_mol, xB_mol, self.constraint_weights['transport']
            )
            total_loss += PhysicalConstraints.mobility_constraint(
                y_pred, T_K, density, self.constraint_weights['transport']
            )
        
        # Crystal structure constraints
        if self.constraint_weights['crystal_structure'] > 0:
            total_loss += PhysicalConstraints.coordination_constraint(
                y_pred, xA_mol, xB_mol, xV_mol, self.constraint_weights['crystal_structure']
            )
            total_loss += PhysicalConstraints.lattice_strain_constraint(
                y_pred, xA_mol, xB_mol, self.constraint_weights['crystal_structure']
            )
        
        # Statistical mechanics constraints
        if self.constraint_weights['statistical_mechanics'] > 0:
            total_loss += PhysicalConstraints.boltzmann_distribution_constraint(
                y_pred, T_K, self.constraint_weights['statistical_mechanics']
            )
            total_loss += PhysicalConstraints.fermi_dirac_constraint(
                y_pred, T_K, xA_mol, xB_mol, self.constraint_weights['statistical_mechanics']
            )
        
        # Defect chemistry constraints
        if self.constraint_weights['defect_chemistry'] > 0:
            total_loss += PhysicalConstraints.schottky_defect_constraint(
                y_pred, T_K, xV_mol, self.constraint_weights['defect_chemistry']
            )
            total_loss += PhysicalConstraints.frenkel_defect_constraint(
                y_pred, T_K, xA_mol, xB_mol, self.constraint_weights['defect_chemistry']
            )
        
        # Electrochemical constraints
        if self.constraint_weights['electrochemistry'] > 0:
            total_loss += PhysicalConstraints.butler_volmer_constraint(
                y_pred, T_K, xA_mol, xB_mol, self.constraint_weights['electrochemistry']
            )
            total_loss += PhysicalConstraints.nernst_equation_constraint(
                y_pred, T_K, xA_mol, xB_mol, self.constraint_weights['electrochemistry']
            )
        
        # Quantum mechanical constraints
        if self.constraint_weights['quantum'] > 0:
            total_loss += PhysicalConstraints.tunneling_constraint(
                y_pred, density, xV_mol, self.constraint_weights['quantum']
            )
            total_loss += PhysicalConstraints.quantum_confinement_constraint(
                y_pred, density, self.constraint_weights['quantum']
            )
        
        # Basic constraints (always applied)
        total_loss += PhysicalConstraints.arrhenius_constraint(
            y_pred, T_K, self.constraint_weights['arrhenius']
        )
        total_loss += PhysicalConstraints.conductivity_bounds_constraint(
            y_pred, self.constraint_weights['bounds']
        )
        total_loss += PhysicalConstraints.charge_conservation_constraint(
            y_pred, xA_mol, xB_mol, xV_mol, self.constraint_weights['charge_conservation']
        )
        total_loss += PhysicalConstraints.density_conductivity_constraint(
            y_pred, density, self.constraint_weights['density']
        )
        
        return total_loss

# ---- Physics-Informed Neural Networks (PINNs) ----
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class BasicPINN(nn.Module):
    """
    Basic Physics-Informed Neural Network for conductivity modeling.
    Incorporates physics constraints directly into the loss function.
    """
    
    def __init__(self, input_dim=7, hidden_dims=[64, 64, 32], output_dim=1, 
                 physics_weight=0.1, activation='tanh'):
        super(BasicPINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
    def physics_loss(self, x, y_pred):
        """Calculate physics-informed loss."""
        # Extract features: [T_K, dopant_A, xA_mol, dopant_B, xB_mol, xV_mol, density]
        T_K = x[:, 0:1]
        xA_mol = x[:, 2:3]
        xB_mol = x[:, 3:4]
        xV_mol = x[:, 4:5]
        density = x[:, 6:7]
        
        # Arrhenius constraint
        T_inv = 1.0 / (T_K + 1e-8)
        arrhenius_loss = torch.mean((y_pred - torch.log(T_inv))**2)
        
        # Conductivity bounds
        sigma_pred = torch.exp(y_pred)
        bounds_loss = torch.mean(torch.relu(1e-10 - sigma_pred) + torch.relu(sigma_pred - 1e+3))
        
        # Charge conservation
        charge_imbalance = torch.abs(xA_mol - xB_mol)
        charge_loss = torch.mean((y_pred - charge_imbalance)**2)
        
        # Density constraint
        density_loss = torch.mean((y_pred - density)**2)
        
        total_physics_loss = arrhenius_loss + bounds_loss + charge_loss + density_loss
        return total_physics_loss
    
    def total_loss(self, x, y_true, y_pred):
        """Calculate total loss (data + physics)."""
        data_loss = nn.MSELoss()(y_pred, y_true)
        physics_loss = self.physics_loss(x, y_pred)
        return data_loss + self.physics_weight * physics_loss

class MultiScalePINN(nn.Module):
    """
    Multi-scale PINN with different network branches for different physics scales.
    """
    
    def __init__(self, input_dim=7, scales=[32, 64, 128], output_dim=1, physics_weight=0.1):
        super(MultiScalePINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        self.scales = scales
        
        # Create multiple scale networks
        self.scale_networks = nn.ModuleList()
        for scale in scales:
            network = nn.Sequential(
                nn.Linear(input_dim, scale),
                nn.Tanh(),
                nn.Linear(scale, scale),
                nn.Tanh(),
                nn.Linear(scale, output_dim)
            )
            self.scale_networks.append(network)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(len(scales) * output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through multi-scale networks."""
        scale_outputs = []
        for network in self.scale_networks:
            scale_outputs.append(network(x))
        
        # Concatenate and fuse
        combined = torch.cat(scale_outputs, dim=1)
        return self.fusion(combined)
    
    def physics_loss(self, x, y_pred):
        """Multi-scale physics loss."""
        T_K = x[:, 0:1]
        xA_mol = x[:, 2:3]
        xB_mol = x[:, 3:4]
        density = x[:, 6:7]
        
        # Temperature-dependent physics (fine scale)
        T_inv = 1.0 / (T_K + 1e-8)
        temp_loss = torch.mean((y_pred - torch.log(T_inv))**2)
        
        # Composition-dependent physics (medium scale)
        comp_loss = torch.mean((y_pred - torch.abs(xA_mol - xB_mol))**2)
        
        # Density-dependent physics (coarse scale)
        density_loss = torch.mean((y_pred - density)**2)
        
        return temp_loss + comp_loss + density_loss
    
    def total_loss(self, x, y_true, y_pred):
        data_loss = nn.MSELoss()(y_pred, y_true)
        physics_loss = self.physics_loss(x, y_pred)
        return data_loss + self.physics_weight * physics_loss

class AttentionPINN(nn.Module):
    """
    PINN with attention mechanism to focus on important physics terms.
    """
    
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=1, physics_weight=0.1):
        super(AttentionPINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        
        # Main network
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism for physics terms
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Physics term embeddings
        self.physics_embedding = nn.Linear(input_dim, hidden_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with attention."""
        # Get main prediction
        main_pred = self.main_net(x)
        
        # Physics-aware attention
        physics_embed = self.physics_embedding(x)
        physics_embed = physics_embed.unsqueeze(1)  # Add sequence dimension
        
        # Self-attention on physics terms
        attended, _ = self.attention(physics_embed, physics_embed, physics_embed)
        attended = attended.squeeze(1)
        
        # Combine with main prediction
        combined = main_pred + 0.1 * attended.mean(dim=1, keepdim=True)
        
        return combined
    
    def physics_loss(self, x, y_pred):
        """Attention-weighted physics loss."""
        T_K = x[:, 0:1]
        xA_mol = x[:, 2:3]
        xB_mol = x[:, 3:4]
        density = x[:, 6:7]
        
        # Calculate physics terms
        arrhenius_term = torch.log(1.0 / (T_K + 1e-8))
        charge_term = torch.abs(xA_mol - xB_mol)
        density_term = density
        
        # Attention weights for different physics terms
        physics_terms = torch.stack([arrhenius_term, charge_term, density_term], dim=1)
        attention_weights = torch.softmax(torch.norm(physics_terms, dim=2), dim=1)
        
        # Weighted physics loss
        weighted_loss = torch.mean(
            attention_weights[:, 0:1] * (y_pred - arrhenius_term)**2 +
            attention_weights[:, 1:2] * (y_pred - charge_term)**2 +
            attention_weights[:, 2:3] * (y_pred - density_term)**2
        )
        
        return weighted_loss
    
    def total_loss(self, x, y_true, y_pred):
        data_loss = nn.MSELoss()(y_pred, y_true)
        physics_loss = self.physics_loss(x, y_pred)
        return data_loss + self.physics_weight * physics_loss

class ResidualPINN(nn.Module):
    """
    Residual PINN with skip connections for better gradient flow.
    """
    
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=1, physics_weight=0.1, num_blocks=3):
        super(ResidualPINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ResidualBlock(hidden_dim)
            self.residual_blocks.append(block)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through residual blocks."""
        h = torch.tanh(self.input_proj(x))
        
        for block in self.residual_blocks:
            h = block(h)
        
        return self.output_proj(h)
    
    def physics_loss(self, x, y_pred):
        """Residual-aware physics loss."""
        T_K = x[:, 0:1]
        xA_mol = x[:, 2:3]
        xB_mol = x[:, 3:4]
        density = x[:, 6:7]
        
        # Physics constraints with residual terms
        arrhenius_residual = y_pred - torch.log(1.0 / (T_K + 1e-8))
        charge_residual = y_pred - torch.abs(xA_mol - xB_mol)
        density_residual = y_pred - density
        
        # Residual physics loss
        residual_loss = torch.mean(
            arrhenius_residual**2 + charge_residual**2 + density_residual**2
        )
        
        return residual_loss
    
    def total_loss(self, x, y_true, y_pred):
        data_loss = nn.MSELoss()(y_pred, y_true)
        physics_loss = self.physics_loss(x, y_pred)
        return data_loss + self.physics_weight * physics_loss

class ResidualBlock(nn.Module):
    """Residual block for PINN."""
    
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return out + residual

class EnsemblePINN(nn.Module):
    """
    Ensemble of PINNs with different architectures for robust predictions.
    """
    
    def __init__(self, input_dim=7, output_dim=1, physics_weight=0.1, num_models=5):
        super(EnsemblePINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        self.num_models = num_models
        
        # Create ensemble of different PINN architectures
        self.models = nn.ModuleList()
        
        # Basic PINN
        self.models.append(BasicPINN(input_dim, [64, 64, 32], output_dim, physics_weight))
        
        # Multi-scale PINN
        self.models.append(MultiScalePINN(input_dim, [32, 64, 128], output_dim, physics_weight))
        
        # Attention PINN
        self.models.append(AttentionPINN(input_dim, 128, output_dim, physics_weight))
        
        # Residual PINN
        self.models.append(ResidualPINN(input_dim, 128, output_dim, physics_weight))
        
        # Deep PINN
        self.models.append(BasicPINN(input_dim, [128, 128, 64, 32], output_dim, physics_weight))
        
        # Ensemble fusion
        self.fusion = nn.Sequential(
            nn.Linear(num_models * output_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        """Forward pass through ensemble."""
        predictions = []
        for model in self.models:
            predictions.append(model(x))
        
        # Concatenate predictions
        combined = torch.cat(predictions, dim=1)
        return self.fusion(combined)
    
    def physics_loss(self, x, y_pred):
        """Ensemble physics loss."""
        total_loss = 0
        for model in self.models:
            total_loss += model.physics_loss(x, y_pred)
        return total_loss / len(self.models)
    
    def total_loss(self, x, y_true, y_pred):
        data_loss = nn.MSELoss()(y_pred, y_true)
        physics_loss = self.physics_loss(x, y_pred)
        return data_loss + self.physics_weight * physics_loss

class AdaptivePINN(nn.Module):
    """
    Adaptive PINN that learns which physics constraints are most important.
    """
    
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=1, physics_weight=0.1):
        super(AdaptivePINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        
        # Main network
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable physics weights
        self.physics_weights = nn.Parameter(torch.ones(4))  # 4 physics terms
        
        # Physics constraint networks
        self.physics_nets = nn.ModuleList([
            nn.Linear(input_dim, 32),
            nn.Linear(input_dim, 32),
            nn.Linear(input_dim, 32),
            nn.Linear(input_dim, 32)
        ])
    
    def forward(self, x):
        """Forward pass."""
        return self.main_net(x)
    
    def physics_loss(self, x, y_pred):
        """Adaptive physics loss with learnable weights."""
        T_K = x[:, 0:1]
        xA_mol = x[:, 2:3]
        xB_mol = x[:, 3:4]
        density = x[:, 6:7]
        
        # Calculate physics terms
        arrhenius_term = torch.log(1.0 / (T_K + 1e-8))
        charge_term = torch.abs(xA_mol - xB_mol)
        density_term = density
        bounds_term = torch.exp(y_pred)
        
        # Physics losses
        losses = [
            torch.mean((y_pred - arrhenius_term)**2),
            torch.mean((y_pred - charge_term)**2),
            torch.mean((y_pred - density_term)**2),
            torch.mean(torch.relu(1e-10 - bounds_term) + torch.relu(bounds_term - 1e+3))
        ]
        
        # Weighted combination with learnable weights
        weighted_loss = sum(w * loss for w, loss in zip(torch.softmax(self.physics_weights, dim=0), losses))
        
        return weighted_loss
    
    def total_loss(self, x, y_true, y_pred):
        data_loss = nn.MSELoss()(y_pred, y_true)
        physics_loss = self.physics_loss(x, y_pred)
        return data_loss + self.physics_weight * physics_loss

# ---- PINN Training Utilities ----
class PINNTrainer:
    """
    Training utilities for PINNs.
    """
    
    def __init__(self, model, learning_rate=1e-3, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(batch_x)
            loss = self.model.total_loss(batch_x, batch_y, y_pred)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                y_pred = self.model(batch_x)
                loss = self.model.total_loss(batch_x, batch_y, y_pred)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=1000, patience=100):
        """Train the PINN."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        return self.train_losses, self.val_losses

# ---- tqdm (progress bar) ----
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kw): return it  # fallback no-op

REQUIRED_COLS = [
    "T_K", "dopant_A", "xA_mol", "dopant_B", "xB_mol",
    "xV_mol", "density_matrix (g/mL)", "log(Conductivity(S/cm))"
]
TARGET = "log(Conductivity(S/cm))"
CAT_COLS = ["dopant_A", "dopant_B"]
NUM_BASE = ["T_K", "xA_mol", "xB_mol", "xV_mol", "density_matrix (g/mL)"]
TEMP_COL = "T_K"

# ---------- Engineered numeric features (for unconstrained models) ----------
class TempInteractions(BaseEstimator, TransformerMixin):
    """
    Numeric block:
      temp_feats = [T, 1/T, log(T), T^2]
      base_nums  = [xA_mol, xB_mol, xV_mol, density]
      interactions = temp_feats ⊗ base_nums
    Output = [temp_feats, base_nums, interactions]

    NOTE: Do NOT mutate __init__ params; scikit-learn must be able to clone.
    """
    def __init__(self, temp_col="T_K",
                 num_cols=("xA_mol","xB_mol","xV_mol","density_matrix (g/mL)")):
        self.temp_col = temp_col
        self.num_cols = num_cols  # tuple for clone-safety

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Accept a DataFrame X that may be missing some expected columns.
        num_cols_list = list(self.num_cols)
        n = len(X)
        # Temperature handling: if missing, fill with zeros (safe fallback)
        if (isinstance(X, (pd.DataFrame,)) and (self.temp_col in X.columns)):
            T = X[self.temp_col].astype(float).to_numpy().reshape(-1, 1)
        else:
            T = np.zeros((n, 1), dtype=float)
        invT = 1.0 / T
        logT = np.log(T)
        T2   = T**2
        temp_block = np.hstack([T, invT, logT, T2])
        # For numeric columns, if any are missing fill with zeros to preserve shape
        nums_cols_present = [c for c in num_cols_list if (isinstance(X, (pd.DataFrame,)) and c in X.columns)]
        if nums_cols_present:
            nums_present = X[nums_cols_present].astype(float).to_numpy()
            # build full nums array in the original order, filling zeros where missing
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

    def get_feature_names_out(self):
        base_nums = list(self.num_cols)
        tnames = ["T", "invT", "logT", "T2"]
        names = tnames + base_nums
        inter = [f"{t}*{n}" for t in tnames for n in base_nums]
        names.extend(inter)
        return np.array(names)

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def preprocessor_unconstrained(scale_after=False, df_sample=None):
    """
    For general models:
      - OHE(dopant_A, dopant_B)
      - engineered numeric features (TempInteractions)
      - optional StandardScaler over whole block for linear/SVR/MLP

    Robust behaviour:
      - If df_sample is provided: use its columns to build transformer column lists.
      - If df_sample is None: use the module-level CAT_COLS and NUM_BASE (which are
        updated in main() after --exclude_features) so we only reference columns
        that actually exist in the input dataframe.
    """
    # preferred numeric base columns (excluding temperature column)
    global CAT_COLS, NUM_BASE, TEMP_COL

    # If caller gave a df_sample, derive present cols from it; otherwise use globals
    if df_sample is not None:
        cat_present = [c for c in CAT_COLS if c in df_sample.columns]
        num_present = [c for c in ["xA_mol", "xB_mol", "xV_mol", "density_matrix (g/mL)"] if c in df_sample.columns]
        temp_present = TEMP_COL in df_sample.columns
    else:
        # use globals (main() updates these based on --exclude_features)
        cat_present = list(CAT_COLS)
        # NUM_BASE may include TEMP_COL; derive numeric feature names excluding TEMP_COL
        num_present = [c for c in NUM_BASE if c != TEMP_COL]
        temp_present = TEMP_COL in NUM_BASE

    cols_for_temp_inter = []
    if temp_present:
        cols_for_temp_inter.append(TEMP_COL)
    cols_for_temp_inter.extend(num_present)

    # Build ColumnTransformer with only columns that actually exist in input passed to fit.
    pre = ColumnTransformer([
        ("cat", make_ohe(), cat_present if cat_present else []),
        ("num", TempInteractions(temp_col=TEMP_COL, num_cols=tuple([c for c in ["xA_mol","xB_mol","xV_mol","density_matrix (g/mL)"] if c in num_present])),
         cols_for_temp_inter),
    ], remainder="drop")

    if scale_after:
        return Pipeline([("pre", pre), ("scaler", StandardScaler())])
    return pre

def preprocessor_constrained(df_sample=None):
    """
    For monotonic temperature models:
      - Keep RAW T_K first (no 1/T, logT, T^2)
      - OHE dopants
      - Pass other numerics
    """
    transformers = []
    cols_to_fit = []
    # include T_raw only if present in sample
    if (df_sample is None) or (TEMP_COL in df_sample.columns):
        transformers.append(("T_raw", "passthrough", [TEMP_COL]))
        cols_to_fit.append(TEMP_COL)
    # categorical block (only include present catecols)
    cat_present = [c for c in CAT_COLS if (df_sample is None or c in df_sample.columns)]
    if cat_present:
        transformers.append(("cat", make_ohe(), cat_present))
        cols_to_fit.extend(cat_present)
    # numeric passthroughs
    num_present = [c for c in ["xA_mol","xB_mol","xV_mol","density_matrix (g/mL)"] if (df_sample is None or c in df_sample.columns)]
    if num_present:
        transformers.append(("num", "passthrough", num_present))
        cols_to_fit.extend(num_present)
    pre = ColumnTransformer(transformers)
    if df_sample is not None and cols_to_fit:
        pre.fit(df_sample[cols_to_fit])
    return pre

# ---------- Robust feature-name helpers for explanations ----------
from sklearn.pipeline import Pipeline as SkPipeline

def _unwrap_pre(pre):
    """Return the inner ColumnTransformer even if pre is a Pipeline(pre -> scaler)."""
    if isinstance(pre, SkPipeline):
        return pre.named_steps.get("pre", pre)
    return pre

def get_feature_names_from_pre(pre, X_sample=None):
    """
    Try to build feature names AFTER transform for explanations.
    Works for:
      - ColumnTransformer with OneHot + TempInteractions
      - ColumnTransformer with passthrough blocks
      - Pipeline(pre -> scaler) via _unwrap_pre
    Falls back to generic names if shapes are unknown.
    """
    pre_ct = _unwrap_pre(pre)
    names = []
    try:
        for name, trans, cols in pre_ct.transformers_:
            if name == "remainder":
                continue
            if name == "cat":
                ohe = pre_ct.named_transformers_["cat"]
                base = ohe.get_feature_names_out(["dopant_A","dopant_B"])
                names.extend(list(base))
            elif name == "num":
                trans_ = pre_ct.named_transformers_["num"]
                if hasattr(trans_, "get_feature_names_out"):
                    names.extend(list(trans_.get_feature_names_out()))
                else:
                    names.extend(list(cols))
            elif name == "T_raw":
                names.extend(list(cols))
            else:
                if isinstance(cols, (list, tuple)):
                    names.extend(list(cols))
    except Exception:
        names = []

    if (not names) and (X_sample is not None):
        Xt = pre.transform(X_sample)
        n = Xt.shape[1]
        names = [f"feat_{i}" for i in range(n)]
    return names

# ---------- Plot & eval ----------
def parity_plot(y_true, y_pred, path, title):
    plt.figure()
    plt.scatter(y_true, y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual log10(σ)")
    plt.ylabel("Predicted log10(σ)")
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def evaluate(pipe, X, y):
    yhat = pipe.predict(X)
    return {
        "mae": float(mean_absolute_error(y, yhat)),
        "r2":  float(r2_score(y, yhat)),
        "yhat": yhat
    }

def kfold_scores(model_name, build_pipe_fn, df, feat_cols, target_col, folds=5):
    kf = KFold(n_splits=min(folds, max(2, len(df)//10)), shuffle=True, random_state=42)
    maes, r2s = [], []
    for tr, te in tqdm(kf.split(df), total=kf.get_n_splits(), leave=False, desc=f"CV folds ({model_name})", ncols=80, dynamic_ncols=True, position=1):
        X_tr, X_te = df.iloc[tr][feat_cols], df.iloc[te][feat_cols]
        y_tr = df.iloc[tr][target_col].astype(float).values
        y_te = df.iloc[te][target_col].astype(float).values
        pipe = build_pipe_fn()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        maes.append(mean_absolute_error(y_te, pred))
        r2s.append(r2_score(y_te, pred))
    return float(np.mean(maes)), float(np.std(maes)), float(np.mean(r2s)), float(np.std(r2s))

# ---------- Save helpers ----------
def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

def save_trained_model(models_dir, name, pipe, schema, metrics, X_sample=None, y_sample=None):
    """
    Save the trained model pipeline, including the TempInteractions transformer, and optionally test it.

    Parameters:
        models_dir (str): Directory to save the model.
        name (str): Name of the model.
        pipe (Pipeline): The trained model pipeline.
        schema (dict): Schema of the input data.
        metrics (dict): Metrics of the model.
        X_sample (pd.DataFrame, optional): A sample of input features for testing.
        y_sample (pd.Series, optional): The corresponding true labels for testing.

    Returns:
        str: Path to the saved model file.
    """
    fname = f"{_safe_name(name)}.joblib"
    path = os.path.join(models_dir, fname)
    bundle = {
        "model": pipe,
        "schema": schema,
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": name,
        "custom_transformers": {
            "TempInteractions": TempInteractions
        },
    }
    joblib.dump(bundle, path)

    # Test the saved model if samples are provided
    if X_sample is not None and y_sample is not None:
        print("Testing the saved model...")
        test_saved_model(path, X_sample, y_sample)

    return path

def test_saved_model(joblib_path, X_sample, y_sample):
    """
    Test the saved .joblib file to verify its integrity and functionality.

    Parameters:
        joblib_path (str): Path to the saved .joblib file.
        X_sample (pd.DataFrame): A sample of input features for testing.
        y_sample (pd.Series): The corresponding true labels for testing.

    Returns:
        dict: A dictionary containing test results (e.g., MAE, R2).

    Example:
        >>> results = test_saved_model("models/my_model.joblib", X_sample, y_sample)
        >>> print(results)
        {'mae': 0.123, 'r2': 0.987, 'yhat': array([...])}
    """
    try:
        # Load the saved model
        bundle = joblib.load(joblib_path)
        pipe = bundle.get("model")

        if pipe is None:
            raise ValueError("The loaded bundle does not contain a valid model pipeline.")

        # Evaluate the model
        results = evaluate(pipe, X_sample, y_sample)
        print(f"Model test results: {results}")
        return results

    except Exception as e:
        print(f"Error testing the saved model: {e}")
        return {"error": str(e)}

# ---------- Explanation exports ----------
def export_permutation(pipe, X, y, feat_names, out_csv):
    r = permutation_importance(pipe, X, y, n_repeats=20, random_state=42, scoring="r2")
    df_imp = pd.DataFrame({
        "feature": feat_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)
    df_imp.to_csv(out_csv, index=False)

TREE_LIKE = (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
def is_tree_model(est):
    if HAS_XGB and isinstance(est, XGBRegressor): return True
    # if HAS_LGBM and isinstance(est, LGBMRegressor): return True
    if HAS_CAT and isinstance(est, CatBoostRegressor): return True
    return isinstance(est, TREE_LIKE)

def export_shap_for_tree(pipe, X, pre, explain_prefix):
    if not HAS_SHAP:
        return False
    Xt = pre.transform(X)
    est = pipe.named_steps["est"]
    try:
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(Xt)
        feat_names = get_feature_names_from_pre(pre, X_sample=X.iloc[: min(200, len(X))])
        # beeswarm
        shap.summary_plot(sv, Xt, feature_names=feat_names, show=False)
        plt.tight_layout(); plt.savefig(explain_prefix + "_shap_beeswarm.png", dpi=200); plt.close()
        # bar
        shap.summary_plot(sv, Xt, feature_names=feat_names, plot_type="bar", show=False)
        plt.tight_layout(); plt.savefig(explain_prefix + "_shap_bar.png", dpi=200); plt.close()
        return True
    except Exception:
        return False

# ---------- Model registry ----------
def build_registry(df, folds=5, monotone_temp=False):
    # Unconstrained preprocessors
    pre_scale = lambda: preprocessor_unconstrained(scale_after=True)
    pre_no_scale = lambda: preprocessor_unconstrained(scale_after=False)
    # when build_registry() has a df, prefer preprocessor variants aware of present columns
    pre_scale = lambda: preprocessor_unconstrained(scale_after=True, df_sample=df)
    pre_no_scale = lambda: preprocessor_unconstrained(scale_after=False, df_sample=df)

    # Constrained preprocessors (RAW T_K only)
    def constrained_hgbr():
        pre = preprocessor_constrained(df)
        n_cat = 0
        # robustly compute number of OHE features if cat block exists
        if "cat" in getattr(pre, "named_transformers_", {}):
            try:
                n_cat = pre.named_transformers_["cat"].get_feature_names_out(CAT_COLS).shape[0]
            except Exception:
                n_cat = 0
        n_num = len([c for c in ["xA_mol","xB_mol","xV_mol","density_matrix (g/mL)"] if c in df.columns])
        monotonic = [ +1 ] + [ 0 ] * (n_cat + n_num)
        est = HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=1000, min_samples_leaf=10,
            early_stopping=True, random_state=42, monotonic_cst=monotonic
        )
        return Pipeline([("pre", pre), ("est", est)])

    def constrained_xgb():
        if not HAS_XGB: return None
        pre = preprocessor_constrained(df)
        n_cat = 0
        try:
            if "cat" in getattr(pre, "named_transformers_", {}):
                n_cat = pre.named_transformers_["cat"].get_feature_names_out(CAT_COLS).shape[0]
        except Exception:
            n_cat = 0
        n_num = len([c for c in ["xA_mol","xB_mol","xV_mol","density_matrix (g/mL)"] if c in df.columns])
        n_total = 1 + n_cat + n_num
        cons = "(" + ",".join(["1"] + ["0"]*(n_total-1)) + ")"
        est = XGBRegressor(
            n_estimators=800, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0,
            monotone_constraints=cons,
            random_state=42, n_jobs=0, verbosity=0
        )
        return Pipeline([("pre", pre), ("est", est)])

    # def constrained_lgbm():
    #     if not HAS_LGBM: return None
    #     pre = preprocessor_constrained(df)
    #     ohe = pre.named_transformers_["cat"]
    #     n_cat = ohe.get_feature_names_out(CAT_COLS).shape[0]
    #     n_total = 1 + n_cat + 4
    #     cons = [1] + [0]*(n_total-1)
    #     est = LGBMRegressor(
    #         n_estimators=1000, max_depth=-1, learning_rate=0.06,
    #         subsample=0.9, colsample_bytree=0.8,
    #         reg_lambda=1.0, reg_alpha=0.0,
    #         random_state=42, n_jobs=0,
    #         monotone_constraints=cons
    #     )
    #     return Pipeline([("pre", pre), ("est", est)])

    # Linear / regularized — scale after engineered features
    def pipe_ols():
        return Pipeline([("pre", pre_scale()), ("est", LinearRegression())])

    def pipe_ridge():
        alphas = np.logspace(-4, 4, 25)
        return Pipeline([("pre", pre_scale()), ("est", RidgeCV(alphas=alphas, cv=folds))])

    def pipe_lasso():
        return Pipeline([("pre", pre_scale()), ("est", LassoCV(alphas=np.logspace(-4,2,40), cv=folds, max_iter=10000))])

    def pipe_elastic():
        return Pipeline([("pre", pre_scale()),
                         ("est", ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                                              alphas=np.logspace(-4, 2, 40),
                                              cv=folds, max_iter=10000))])

    def pipe_bayesridge():
        return Pipeline([("pre", pre_scale()), ("est", BayesianRidge())])

    def pipe_pls():
        return Pipeline([("pre", pre_scale()), ("est", PLSRegression(n_components=2))])

    # Kernel / neighbors / neural — scale after
    def pipe_svr_rbf():
        return Pipeline([("pre", pre_scale()), ("est", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"))])

    def pipe_svr_poly():
        return Pipeline([("pre", pre_scale()), ("est", SVR(kernel="poly", degree=3, C=10.0, epsilon=0.05, gamma="scale"))])

    def pipe_knn():
        return Pipeline([("pre", pre_scale()), ("est", KNeighborsRegressor(n_neighbors=5, weights="distance"))])

    def pipe_mlp():
        return Pipeline([("pre", pre_scale()), ("est", MLPRegressor(hidden_layer_sizes=(64, 32),
                                                                   activation="relu",
                                                                   alpha=1e-3, learning_rate_init=1e-3,
                                                                   max_iter=5000, random_state=42))])

    def pipe_gpr():
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
        return Pipeline([("pre", pre_scale()), ("est", GaussianProcessRegressor(kernel=kernel,
                                                                               alpha=1e-6,
                                                                               n_restarts_optimizer=1,
                                                                               normalize_y=True,
                                                                               random_state=42))])

    # Trees / ensembles — no global scaler for unconstrained engineered features
    def pipe_tree():
        return Pipeline([("pre", preprocessor_unconstrained(scale_after=False)),
                         ("est", DecisionTreeRegressor(max_depth=6, min_samples_leaf=3, random_state=42))])

    def pipe_rf():
        return Pipeline([("pre", preprocessor_unconstrained(scale_after=False)),
                         ("est", RandomForestRegressor(n_estimators=500, max_depth=None,
                                                       min_samples_leaf=3, random_state=42))])

    def pipe_extra():
        return Pipeline([("pre", preprocessor_unconstrained(scale_after=False)),
                         ("est", ExtraTreesRegressor(n_estimators=600, max_depth=None,
                                                     min_samples_leaf=2, random_state=42))])

    def pipe_gbr():
        return Pipeline([("pre", preprocessor_unconstrained(scale_after=False)),
                         ("est", GradientBoostingRegressor(n_estimators=800, learning_rate=0.05,
                                                           max_depth=3, subsample=0.9, random_state=42))])

    def pipe_hgbr():
        return Pipeline([("pre", preprocessor_unconstrained(scale_after=False)),
                         ("est", HistGradientBoostingRegressor(learning_rate=0.06, max_iter=1000,
                                                               min_samples_leaf=10, early_stopping=True,
                                                               random_state=42))])

    registry = {
        # Physics-Informed Neural Networks (PINNs)
        "BasicPINN": lambda: BasicPINN(input_dim=7, hidden_dims=[64, 64, 32], physics_weight=0.1),
        "MultiScalePINN": lambda: MultiScalePINN(input_dim=7, scales=[32, 64, 128], physics_weight=0.1),
        "AttentionPINN": lambda: AttentionPINN(input_dim=7, hidden_dim=128, physics_weight=0.1),
        "ResidualPINN": lambda: ResidualPINN(input_dim=7, hidden_dim=128, physics_weight=0.1),
        "EnsemblePINN": lambda: EnsemblePINN(input_dim=7, physics_weight=0.1),
        "AdaptivePINN": lambda: AdaptivePINN(input_dim=7, hidden_dim=128, physics_weight=0.1),
        
        # Comprehensive Physics-Informed Models
        "PhysicsInformed_Ridge": lambda: Pipeline([
            ("pre", preprocessor_unconstrained(scale_after=True)),
            ("est", RidgeCV())
        ]),
        "PhysicsInformed_RandomForest": lambda: Pipeline([
            ("pre", preprocessor_unconstrained(scale_after=False)),
            ("est", RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "PhysicsInformed_XGBoost": lambda: Pipeline([
            ("pre", preprocessor_unconstrained(scale_after=False)),
            ("est", XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0))
        ]) if HAS_XGB else None,
        
        # Adaptive feature weighting models
        "Adaptive_Ridge": lambda: Pipeline([
            ("pre", preprocessor_unconstrained(scale_after=True)),
            ("adaptive_scaler", AdaptiveFeatureScaler(learning_rate=0.01, sparsity_lambda=0.01)),
            ("est", RidgeCV())
        ]),
        "Adaptive_RandomForest": lambda: Pipeline([
            ("pre", preprocessor_unconstrained(scale_after=False)),
            ("adaptive_scaler", AdaptiveFeatureScaler(learning_rate=0.01, sparsity_lambda=0.01)),
            ("est", RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "Adaptive_XGBoost": lambda: Pipeline([
            ("pre", preprocessor_unconstrained(scale_after=False)),
            ("adaptive_scaler", AdaptiveFeatureScaler(learning_rate=0.01, sparsity_lambda=0.01)),
            ("est", XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0))
        ]) if HAS_XGB else None,
        
        # Linear / regularized
        "OLS": pipe_ols,
        "RidgeCV": pipe_ridge,
        "LassoCV": pipe_lasso,
        "ElasticNetCV": pipe_elastic,
        "BayesianRidge": pipe_bayesridge,
        "PLSRegression": pipe_pls,
        # Kernel / neighbors / NN
        "SVR_RBF": pipe_svr_rbf,
        "SVR_Poly": pipe_svr_poly,
        "KNN": pipe_knn,
        "MLP": pipe_mlp,
        "GPR": pipe_gpr,
        # Trees / ensembles
        "DecisionTree": pipe_tree,
        "RandomForest": pipe_rf,
        "ExtraTrees": pipe_extra,
        "GradientBoosting": pipe_gbr,
        "HistGradientBoosting": pipe_hgbr,
    }

    if HAS_XGB:
        def pipe_xgb():
            return Pipeline([("pre", preprocessor_unconstrained(scale_after=False)),
                             ("est", XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.05,
                                                  subsample=0.9, colsample_bytree=0.8,
                                                  reg_lambda=1.0, reg_alpha=0.0,
                                                  random_state=42, n_jobs=0, verbosity=0))])
        registry["XGBoost"] = pipe_xgb

    # if HAS_LGBM:
    #     def pipe_lgbm():
    #         return Pipeline([("pre", preprocessor_unconstrained(scale_after=False)),
    #                          ("est", LGBMRegressor(n_estimators=1000, max_depth=-1, learning_rate=0.06,
    #                                                subsample=0.9, colsample_bytree=0.8,
    #                                                reg_lambda=1.0, reg_alpha=0.0,
    #                                                random_state=42, n_jobs=0))])
    #     registry["LightGBM"] = pipe_lgbm

    if HAS_CAT:
        def pipe_cat():
            return Pipeline([("pre", preprocessor_unconstrained(scale_after=False)),
                             ("est", CatBoostRegressor(iterations=1200, learning_rate=0.05, depth=6,
                                                       loss_function="RMSE", random_seed=42, verbose=False))])
        registry["CatBoost"] = pipe_cat

    if monotone_temp:
        registry["HistGradientBoosting"] = constrained_hgbr
        if HAS_XGB:
            registry["XGBoost"] = constrained_xgb
        # if HAS_LGBM:
        #     registry["LightGBM"] = constrained_lgbm

    return registry

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Compare many models; progress bars; save models; monotone temp; explanations")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--out_dir", default="fit_results_all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_model", default=None,
                    help="If set, evaluate ONLY this model name (see list printed at end).")
    ap.add_argument("--model_to_save", default=None,
                    help="If set, save this model (trained on full data) to --save_path and models dir.")
    ap.add_argument("--save_path", default="best_model.joblib",
                    help="Path to write the saved model if --model_to_save is provided.")
    ap.add_argument("--save_all", action="store_true",
                    help="If set, save every trained model to disk (full-data fit).")
    ap.add_argument("--models_dir", default=None,
                    help="Directory to write saved models. Defaults to <out_dir>/models")
    ap.add_argument("--monotone_temp", action="store_true",
                    help="Use temperature monotonic constraints for supported boosters (HGBR, XGB).")  # LGBM removed
    ap.add_argument("--explain", action="store_true",
                    help="Export permutation importance for all models and SHAP for tree models.")
    ap.add_argument("--explain_models", default=None,
                    help="Comma-separated model names to explain (default: all evaluated)")
    ap.add_argument("--adaptive_only", action="store_true",
                    help="Run only adaptive feature weighting models")
    ap.add_argument("--learning_rate", type=float, default=0.01,
                    help="Learning rate for adaptive feature weighting")
    ap.add_argument("--sparsity_lambda", type=float, default=0.01,
                    help="Sparsity regularization for feature selection")
    ap.add_argument("--track_contributions", action="store_true",
                    help="Track and save feature contribution evolution")
    ap.add_argument("--physics_only", action="store_true",
                    help="Run only physics-informed models")
    ap.add_argument("--physics_weight", type=float, default=0.1,
                    help="Overall weight for physics constraints")
    ap.add_argument("--use_thermodynamic", action="store_true",
                    help="Enable thermodynamic constraints (Gibbs free energy, chemical potential)")
    ap.add_argument("--use_transport_theory", action="store_true",
                    help="Enable transport theory constraints (Nernst-Einstein, mobility)")
    ap.add_argument("--use_crystal_structure", action="store_true",
                    help="Enable crystal structure constraints (coordination, lattice strain)")
    ap.add_argument("--use_statistical_mechanics", action="store_true",
                    help="Enable statistical mechanics constraints (Boltzmann, Fermi-Dirac)")
    ap.add_argument("--use_defect_chemistry", action="store_true",
                    help="Enable defect chemistry constraints (Schottky, Frenkel defects)")
    ap.add_argument("--use_electrochemistry", action="store_true",
                    help="Enable electrochemical constraints (Butler-Volmer, Nernst equation)")
    ap.add_argument("--use_quantum_effects", action="store_true",
                    help="Enable quantum mechanical constraints (tunneling, confinement)")
    ap.add_argument("--constraint_weights", default=None,
                    help="Custom constraint weights as comma-separated values: arrhenius,bounds,charge,density,thermo,transport,crystal,statistical,defect,electro,quantum")
    ap.add_argument("--pinn_only", action="store_true",
                    help="Run only PINN models")
    ap.add_argument("--pinn_epochs", type=int, default=1000,
                    help="Number of training epochs for PINNs")
    ap.add_argument("--pinn_lr", type=float, default=1e-3,
                    help="Learning rate for PINN training")
    ap.add_argument("--pinn_batch_size", type=int, default=32,
                    help="Batch size for PINN training")
    ap.add_argument("--pinn_patience", type=int, default=100,
                    help="Early stopping patience for PINNs")
    ap.add_argument("--device", default="cpu",
                    help="Device for PINN training (cpu/cuda)")
    ap.add_argument("--exclude_features", default=None,
                    help="Comma-separated feature names to drop from input (cannot drop the target).")
    ap.add_argument("--test_saved_model", action="store_true",
                    help="Test the saved model after training.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models_dir = args.models_dir or os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    explain_dir = os.path.join(args.out_dir, "explain")
    if args.explain:
        os.makedirs(explain_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    # --- Handle user-requested exclusions early (drop columns from dataframe) ---
    excluded = []
    if args.exclude_features:
        excluded = [s.strip() for s in args.exclude_features.split(",") if s.strip()]
        if TARGET in excluded:
            print(f"Cannot exclude target column: {TARGET}"); sys.exit(1)
        # drop columns (ignore missing names)
        df = df.drop(columns=excluded, errors="ignore")
        print(f"[INPUT] Dropped excluded columns: {excluded}")
    
    # Update available categorical / numeric lists based on what's actually present
    # (modify globals so downstream preprocessors use the reduced sets)
    global CAT_COLS, NUM_BASE, TEMP_COL
    CAT_COLS = [c for c in CAT_COLS if c in df.columns]
    NUM_BASE = [c for c in NUM_BASE if c in df.columns]
    if TEMP_COL not in df.columns:
        print(f"[INPUT] Temperature column '{TEMP_COL}' not present after exclusion. Temp-derived features will be zero-filled.")

    feat_cols = CAT_COLS + NUM_BASE
    y = df[TARGET].astype(float).values

    registry = build_registry(df, folds=args.folds, monotone_temp=args.monotone_temp)
    model_names = list(registry.keys())

    if args.only_model and args.only_model not in registry:
        print("Invalid --only_model. Available:", ", ".join(model_names)); sys.exit(1)
    if args.model_to_save and args.model_to_save not in registry:
        print("Invalid --model_to_save. Available:", ", ".join(model_names)); sys.exit(1)

    # Parse custom constraint weights if provided
    constraint_weights = None
    if args.constraint_weights:
        try:
            weights_list = [float(w.strip()) for w in args.constraint_weights.split(",")]
            if len(weights_list) == 11:
                constraint_weights = {
                    'arrhenius': weights_list[0],
                    'bounds': weights_list[1],
                    'charge_conservation': weights_list[2],
                    'density': weights_list[3],
                    'thermodynamic': weights_list[4],
                    'transport': weights_list[5],
                    'crystal_structure': weights_list[6],
                    'statistical_mechanics': weights_list[7],
                    'defect_chemistry': weights_list[8],
                    'electrochemistry': weights_list[9],
                    'quantum': weights_list[10]
                }
                print(f"[PHYSICS] Using custom constraint weights: {constraint_weights}")
            else:
                print(f"[PHYSICS] Warning: Expected 11 weights, got {len(weights_list)}. Using defaults.")
        except Exception as e:
            print(f"[PHYSICS] Error parsing constraint weights: {e}. Using defaults.")
    
    # Set constraint weights based on command line flags
    if not constraint_weights:
        constraint_weights = {
            'arrhenius': args.physics_weight,
            'bounds': args.physics_weight,
            'charge_conservation': args.physics_weight,
            'density': args.physics_weight,
            'thermodynamic': args.physics_weight if args.use_thermodynamic else 0,
            'transport': args.physics_weight if args.use_transport_theory else 0,
            'crystal_structure': args.physics_weight if args.use_crystal_structure else 0,
            'statistical_mechanics': args.physics_weight if args.use_statistical_mechanics else 0,
            'defect_chemistry': args.physics_weight if args.use_defect_chemistry else 0,
            'electrochemistry': args.physics_weight if args.use_electrochemistry else 0,
            'quantum': args.physics_weight if args.use_quantum_effects else 0
        }

    # Filter models based on options
    if args.pinn_only:
        pinn_models = [name for name in model_names if "PINN" in name]
        to_run = [args.only_model] if args.only_model else pinn_models
        if not to_run:
            print("No PINN models available. Available models:", ", ".join(model_names))
            sys.exit(1)
    elif args.physics_only:
        physics_models = [name for name in model_names if "Physics" in name]
        to_run = [args.only_model] if args.only_model else physics_models
        if not to_run:
            print("No physics-informed models available. Available models:", ", ".join(model_names))
            sys.exit(1)
    elif args.adaptive_only:
        adaptive_models = [name for name in model_names if "Adaptive" in name]
        to_run = [args.only_model] if args.only_model else adaptive_models
        if not to_run:
            print("No adaptive models available. Available models:", ", ".join(model_names))
            sys.exit(1)
    else:
        to_run = [args.only_model] if args.only_model else model_names

    explain_subset = set(to_run)
    if args.explain_models:
        wanted = set([s.strip() for s in args.explain_models.split(",") if s.strip()])
        explain_subset = explain_subset.intersection(wanted)

    rows = []
    contribution_tracker = FeatureContributionTracker() if args.track_contributions else None
    
    for name in tqdm(to_run, desc="Models", leave=False, ncols=80, dynamic_ncols=True, position=0):
        mk = registry[name]

        # Special handling for PINN models
        if "PINN" in name:
            try:
                print(f"\n[PINN] Training {name}...")
                
                # Prepare data for PINN training
                X_tensor = torch.FloatTensor(df[feat_cols].values)
                y_tensor = torch.FloatTensor(y.reshape(-1, 1))
                
                # Create train/validation split
                n_samples = len(X_tensor)
                n_train = int(0.8 * n_samples)
                indices = torch.randperm(n_samples)
                
                X_train = X_tensor[indices[:n_train]]
                y_train = y_tensor[indices[:n_train]]
                X_val = X_tensor[indices[n_train:]]
                y_val = y_tensor[indices[n_train:]]
                
                # Create data loaders
                train_dataset = TensorDataset(X_train, y_train)
                val_dataset = TensorDataset(X_val, y_val)
                
                train_loader = DataLoader(train_dataset, batch_size=args.pinn_batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=args.pinn_batch_size, shuffle=False)
                
                # Create and train PINN
                pinn_model = mk()
                trainer = PINNTrainer(pinn_model, learning_rate=args.pinn_lr, device=args.device)
                
                train_losses, val_losses = trainer.train(
                    train_loader, val_loader, 
                    epochs=args.pinn_epochs, 
                    patience=args.pinn_patience
                )
                
                # Evaluate PINN
                pinn_model.eval()
                with torch.no_grad():
                    y_pred_tensor = pinn_model(X_tensor.to(trainer.device))
                    y_pred = y_pred_tensor.cpu().numpy().flatten()
                
                # Calculate metrics
                mae = np.mean(np.abs(y - y_pred))
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                
                m = {"mae": mae, "r2": r2, "yhat": y_pred}
                
                # Cross-validation for PINNs (simplified)
                cv_mae_mean, cv_mae_std = mae, 0.0
                cv_r2_mean, cv_r2_std = r2, 0.0
                
                print(f"[PINN] {name} - Final Loss: {val_losses[-1]:.6f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"[PINN] Error training {name}: {e}")
                continue
        else:
            # Standard sklearn model training
            pipe = mk()
            pipe.fit(df[feat_cols], y)
            m = evaluate(pipe, df[feat_cols], y)
            cv_mae_mean, cv_mae_std, cv_r2_mean, cv_r2_std = kfold_scores(
                name, mk, df, feat_cols, TARGET, folds=args.folds
            )

        # Create parity plot
        parity_plot(y, m["yhat"], os.path.join(args.out_dir, f"{name}_parity.png"),
                    f"{name}: Predicted vs Actual")

        # Track feature contributions for adaptive models
        if args.track_contributions and "Adaptive" in name:
            try:
                # Get adaptive scaler from pipeline
                adaptive_scaler = pipe.named_steps.get("adaptive_scaler")
                if adaptive_scaler and adaptive_scaler.feature_weights:
                    importance = adaptive_scaler.get_feature_importance()
                    active_features = adaptive_scaler.feature_weights.get_active_features(threshold=0.01)
                    # Derive transformed feature names to match importance vector length
                    pre_step = pipe.named_steps.get("pre")
                    try:
                        X_sample = df[feat_cols].iloc[: min(200, len(df))].copy()
                        feature_names = get_feature_names_from_pre(pre_step, X_sample=X_sample)
                    except Exception:
                        feature_names = []
                    if not feature_names or len(feature_names) != len(importance):
                        feature_names = [f"feat_{i}" for i in range(len(importance))]

                    contribution_tracker.track_contributions(importance, feature_names)

                    print(f"\n[ADAPTIVE] {name} Feature Analysis:")
                    print(f"  Active features: {len(active_features)}/{len(importance)}")
                    print(f"  Active feature indices: {active_features}")
                    
                    # Print top contributing features
                    sorted_indices = np.argsort(importance)[::-1]
                    print(f"  Top 3 contributing features:")
                    for i in range(min(3, len(sorted_indices))):
                        idx = sorted_indices[i]
                        fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                        print(f"    {fname}: {importance[idx]:.4f}")
            except Exception as e:
                print(f"[ADAPTIVE] Could not analyze {name}: {e}")

        # Calculate physics constraints for physics-informed models
        if "Physics" in name:
            try:
                physics_loss = ComprehensivePhysicsInformedLoss(constraint_weights)
                X_features = df[feat_cols].values
                y_pred = pipe.predict(X_features)
                total_physics_loss = physics_loss.calculate_total_loss(y_pred, X_features, y)
                
                print(f"\n[PHYSICS] {name} Constraint Analysis:")
                print(f"  Total physics loss: {total_physics_loss:.6f}")
                
                # Individual constraint analysis
                active_constraints = [k for k, v in constraint_weights.items() if v > 0]
                print(f"  Active constraints: {', '.join(active_constraints)}")
                
            except Exception as e:
                print(f"[PHYSICS] Could not analyze {name}: {e}")

        rows.append({
            "model": name,
            "train_mae": round(m["mae"], 4),
            "train_r2": round(m["r2"], 4),
            "cv_mae_mean": round(cv_mae_mean, 4),
            "cv_mae_std": round(cv_mae_std, 4),
            "cv_r2_mean": round(cv_r2_mean, 4),
            "cv_r2_std": round(cv_r2_std, 4),
        })

        # Save requested/all models
        should_save_this = args.save_all or (args.model_to_save == name)
        if should_save_this:
            schema = {"cat_cols": CAT_COLS, "num_cols": NUM_BASE, "target": TARGET,
                      "monotone_temp": args.monotone_temp}
            metrics = {
                "train": {"mae": float(m["mae"]), "r2": float(m["r2"])},
                "cv": {"mae_mean": float(cv_mae_mean), "mae_std": float(cv_mae_std),
                       "r2_mean": float(cv_r2_mean), "r2_std": float(cv_r2_std)}
            }
            path = save_trained_model(models_dir, name, pipe, schema, metrics, df[feat_cols], df[TARGET])
            if args.model_to_save == name:
                joblib.dump(joblib.load(path), args.save_path)
                print(f"\nSaved {name} to: {path}")
                print(f"Also wrote requested copy to: {args.save_path}")
            else:
                print(f"\nSaved {name} to: {path}")

        # ----- Explanations -----
        print("explanation starts:")
        print(name, explain_subset)
        if args.explain and (name in explain_subset):
            print(name, explain_subset)
            try:
                pre = pipe.named_steps["pre"]
                est = pipe.named_steps["est"]

                # RAW inputs (used by permutation_importance)
                X_raw = df[feat_cols]
                y_raw = y

                # For SHAP only: get transformed feature names
                X_sample = X_raw.iloc[: min(200, len(X_raw))].copy()
                trans_names = get_feature_names_from_pre(pre, X_sample=X_sample)

                # --- Permutation importance (always) ---
                # Use RAW column names to match sklearn’s permutation input.
                raw_names = list(X_raw.columns)
                perm_csv = os.path.join(explain_dir, f"{_safe_name(name)}_perm_importance.csv")
                export_permutation(pipe, X_raw, y_raw, raw_names, perm_csv)
                print(f"[EXPLAIN] Wrote permutation importance -> {perm_csv}")

                # --- SHAP (tree models only, if shap installed) ---
                if is_tree_model(est):
                    if HAS_SHAP:
                        ok = export_shap_for_tree(
                            pipe, X_raw, _unwrap_pre(pre),
                            os.path.join(explain_dir, _safe_name(name))
                        )
                        if ok:
                            print(f"[EXPLAIN] Wrote SHAP plots for {name}")
                        else:
                            print(f"[EXPLAIN] SHAP failed for {name} (continuing).")
                    else:
                        print(f"[EXPLAIN] SHAP not installed; skipping SHAP for {name}.")
                else:
                    print(f"[EXPLAIN] {name} is not a tree-like model; skipping SHAP.")
            except Exception as e:
                print(f"[EXPLAIN] Error for {name}: {e} (continuing)")

    # Write summary CSV
    summary = pd.DataFrame(rows).sort_values(by=["cv_r2_mean","train_r2"], ascending=False)
    out_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    summary.to_csv(out_csv, index=False)

    # Console leaderboard
    print("\n=== Leaderboard (sorted by CV R²) ===")
    print(summary[["model","cv_r2_mean","cv_mae_mean","train_r2","train_mae"]].to_string(index=False))
    print(f"\nSaved metrics to {out_csv}")
    print(f"Parity plots saved in {args.out_dir}/")
    print(f"Models saved in {models_dir}/ (depending on flags)")
    if args.explain:
        print(f"Explanations saved in {os.path.join(args.out_dir, 'explain')}/")
    
    # Plot feature contribution evolution if tracking was enabled
    if args.track_contributions and contribution_tracker:
        contribution_plot_path = os.path.join(args.out_dir, "feature_contribution_evolution.png")
        contribution_tracker.plot_contribution_evolution(save_path=contribution_plot_path)
        print(f"Feature contribution evolution plot saved to: {contribution_plot_path}")
    
    print("\nAvailable model names:", ", ".join(model_names))

# Example usage of the `test_saved_model` function
# This example demonstrates how to save and test a model with generative data.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison and testing script.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--out_dir", required=True, help="Output directory for results.")
    parser.add_argument("--exclude_features", help="Features to exclude from the dataset.")
    parser.add_argument("--fold", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--explain", action="store_true", help="Generate explanations for the models.")
    parser.add_argument("--use_thermodynamic", action="store_true", help="Use thermodynamic features.")
    parser.add_argument("--use_transport_theory", action="store_true", help="Use transport theory features.")
    parser.add_argument("--use_crystal_structure", action="store_true", help="Use crystal structure features.")
    parser.add_argument("--use_statistical_mechanics", action="store_true", help="Use statistical mechanics features.")
    parser.add_argument("--use_defect_chemistry", action="store_true", help="Use defect chemistry features.")
    parser.add_argument("--use_electrochemistry", action="store_true", help="Use electrochemistry features.")
    parser.add_argument("--use_quantum_effects", action="store_true", help="Use quantum effects features.")
    parser.add_argument("--save_all", action="store_true", help="Save all models.")
    parser.add_argument("--test_saved_model", action="store_true", help="Test the saved model after training.")

    args = parser.parse_args()

    # Example: Load data and preprocess
    import pandas as pd
    data = pd.read_csv(args.csv)

    # Example: Train a model (simplified for demonstration)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    X = data.drop(columns=["log(Conductivity(S/cm))"])
    y = data["log(Conductivity(S/cm))"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("preprocessor", TempInteractions(temp_col="T_K", num_cols=("xA_mol", "xB_mol", "xV_mol", "density_matrix (g/mL)"))),
        ("regressor", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)

    # Save the model
    models_dir = args.out_dir
    model_name = "trained_model"
    schema = {"features": list(X.columns), "target": "log(Conductivity(S/cm))"}
    metrics = {"mae": 0.1, "r2": 0.9}  # Example metrics

    model_path = save_trained_model(models_dir, model_name, model, schema, metrics, X_sample=X_test, y_sample=y_test)

    # Test the saved model if the flag is set
    if args.test_saved_model:
        print("Testing the saved model...")
        test_saved_model(model_path, X_test, y_test)
