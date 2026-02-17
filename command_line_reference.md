# Complete Command Line Reference for PIML Model Compare Script

## 📋 **Basic Arguments**

### **Required Arguments**
| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--csv` | string | **REQUIRED** - Input CSV file path | `--csv data.csv` |

### **Output & Directory Arguments**
| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--out_dir` | string | `fit_results_all` | Output directory for results | `--out_dir my_results` |
| `--models_dir` | string | `<out_dir>/models` | Directory for saved models | `--models_dir saved_models` |

### **Cross-Validation Arguments**
| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--folds` | int | `5` | Number of CV folds | `--folds 10` |

## 🎯 **Model Selection Arguments**

### **Single Model Execution**
| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--only_model` | string | `None` | Run only this specific model | `--only_model RidgeCV` |

### **Model Categories**
| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--physics_only` | flag | Run only physics-informed models | `--physics_only` |
| `--adaptive_only` | flag | Run only adaptive feature weighting models | `--adaptive_only` |

## 💾 **Model Saving Arguments**

### **Save Options**
| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--save_all` | flag | Save all trained models to disk | `--save_all` |
| `--model_to_save` | string | Save specific model | `--model_to_save RidgeCV` |
| `--save_path` | string | `best_model.joblib` | Path for saved model | `--save_path my_model.joblib` |

## 🔬 **Physics-Informed ML Arguments**

### **Physics Constraint Control**
| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--physics_weight` | float | `0.1` | Overall weight for physics constraints | `--physics_weight 0.2` |
| `--use_thermodynamic` | flag | Enable thermodynamic constraints | `--use_thermodynamic` |
| `--use_transport_theory` | flag | Enable transport theory constraints | `--use_transport_theory` |
| `--use_crystal_structure` | flag | Enable crystal structure constraints | `--use_crystal_structure` |
| `--use_statistical_mechanics` | flag | Enable statistical mechanics constraints | `--use_statistical_mechanics` |
| `--use_defect_chemistry` | flag | Enable defect chemistry constraints | `--use_defect_chemistry` |
| `--use_electrochemistry` | flag | Enable electrochemical constraints | `--use_electrochemistry` |
| `--use_quantum_effects` | flag | Enable quantum mechanical constraints | `--use_quantum_effects` |

### **Custom Constraint Weights**
| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--constraint_weights` | string | `None` | Custom weights for all constraints | `--constraint_weights "0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05"` |

**Constraint Weight Order:**
1. `arrhenius` - Arrhenius temperature dependence
2. `bounds` - Conductivity bounds
3. `charge` - Charge conservation
4. `density` - Density-conductivity relationship
5. `thermo` - Thermodynamic constraints
6. `transport` - Transport theory constraints
7. `crystal` - Crystal structure constraints
8. `statistical` - Statistical mechanics constraints
9. `defect` - Defect chemistry constraints
10. `electro` - Electrochemical constraints
11. `quantum` - Quantum mechanical constraints

## 🧠 **Adaptive Feature Weighting Arguments**

### **Adaptive Learning Parameters**
| Argument | Type | Default | Description | Example |
|----------|------|---------|-------------|---------|
| `--learning_rate` | float | `0.01` | Learning rate for adaptive feature weighting | `--learning_rate 0.02` |
| `--sparsity_lambda` | float | `0.01` | Sparsity regularization for feature selection | `--sparsity_lambda 0.05` |
| `--track_contributions` | flag | Track and visualize feature contribution evolution | `--track_contributions` |

## 🔍 **Analysis & Explanation Arguments**

### **Model Explanation**
| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--explain` | flag | Export permutation importance and SHAP plots | `--explain` |

### **Temperature Constraints**
| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--monotone_temp` | flag | Use temperature monotonic constraints for boosters | `--monotone_temp` |

## 📊 **Available Models**

### **Physics-Informed Models**
- `PhysicsInformed_Ridge`
- `PhysicsInformed_RandomForest`
- `PhysicsInformed_XGBoost` (if XGBoost installed)

### **Adaptive Feature Weighting Models**
- `Adaptive_Ridge`
- `Adaptive_RandomForest`
- `Adaptive_XGBoost` (if XGBoost installed)

### **Linear/Regularized Models**
- `OLS`
- `RidgeCV`
- `LassoCV`
- `ElasticNetCV`
- `BayesianRidge`
- `PLSRegression`

### **Kernel/Neighbors/Neural Models**
- `SVR_RBF`
- `SVR_Poly`
- `KNN`
- `MLP`
- `GPR`

### **Tree/Ensemble Models**
- `DecisionTree`
- `RandomForest`
- `ExtraTrees`
- `GradientBoosting`
- `HistGradientBoosting`
- `XGBoost` (if XGBoost installed)
- `CatBoost` (if CatBoost installed)

## 🚀 **Usage Examples**

### **Basic Usage**
```bash
python model_compare.py --csv data.csv --out_dir results --folds 5
```

### **Physics-Informed Only**
```bash
python model_compare.py --csv data.csv --physics_only --physics_weight 0.2
```

### **All Physics Constraints**
```bash
python model_compare.py --csv data.csv --physics_only \
    --use_thermodynamic --use_transport_theory --use_crystal_structure \
    --use_statistical_mechanics --use_defect_chemistry --use_electrochemistry \
    --use_quantum_effects --physics_weight 0.15
```

### **Adaptive Feature Weighting**
```bash
python model_compare.py --csv data.csv --adaptive_only \
    --track_contributions --learning_rate 0.02 --sparsity_lambda 0.01
```

### **Custom Constraint Weights**
```bash
python model_compare.py --csv data.csv --physics_only \
    --constraint_weights "0.2,0.1,0.15,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05"
```

### **Single Model with Explanations**
```bash
python model_compare.py --csv data.csv --only_model RidgeCV \
    --explain --save_all --out_dir single_model_results
```

### **Comprehensive Analysis**
```bash
python model_compare.py --csv data.csv --out_dir comprehensive \
    --folds 10 --save_all --explain --monotone_temp \
    --physics_weight 0.1 --use_thermodynamic --use_transport_theory \
    --track_contributions
```

## 📝 **Notes**

- **Required**: Only `--csv` is required, all other arguments are optional
- **Flags**: Arguments without values are boolean flags (use or don't use)
- **Model Names**: Use exact model names as shown in the Available Models section
- **File Paths**: Use forward slashes `/` or double backslashes `\\` for Windows paths
- **Constraint Weights**: Must provide exactly 11 comma-separated values
- **Dependencies**: Some models require optional packages (XGBoost, CatBoost, SHAP)

## 🔧 **Troubleshooting**

- **Model Not Found**: Check available model names with `--only_model InvalidModel`
- **Constraint Weight Error**: Ensure exactly 11 comma-separated values
- **File Not Found**: Check CSV file path and permissions
- **Memory Issues**: Reduce `--folds` or use `--only_model` for single models
