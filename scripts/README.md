# Reproducible ML Analysis Scripts

## Overview
This folder contains the Python script to reproducibly generate all machine learning metrics, visualizations, and evaluation results for the food analysis project.

---

## 📜 Scripts

### generate_metrics.py
**Complete reproducible analysis pipeline**

Generates:
- All ML evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Cross-validation scores (5-fold)
- Confidence intervals (95% CI)
- Confusion matrices
- Feature importance rankings
- Publication-quality visualizations (3 PNG files)
- JSON and CSV summaries

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install scikit-learn pandas numpy matplotlib
```

### Run the Analysis
```bash
python3 generate_metrics.py
```

### Expected Output
```
================================================================================
LOADING AND PREPROCESSING DATA
================================================================================
✓ Loaded raw recipes: (83782, 12)
✓ Loaded raw reviews: (731927, 5)
...
✓ Metrics generation complete!
```

**Output files generated**:
- `../data/metrics_summary.json`
- `../data/results_table.csv`
- `../assets/papers/roc_curves.png`
- `../assets/papers/learning_curves.png`
- `../assets/papers/feature_importance.png`

---

## 📊 What the Script Does

### Step 1: Data Loading & Preprocessing
- Loads RAW_recipes.csv and interactions.csv
- Cleans missing values
- Parses nutrition arrays into individual features
- Calculates nutrition proportion features (as % of calories)
- Extracts cuisine tags (American vs European)

**Data handling**:
- Missing values: Replaced 0 with NaN (assumed missing)
- Dates: Converted to datetime format
- Text: Custom token pattern for ingredient parsing

### Step 2: Feature Engineering
- **Baseline features** (3): Calories, sugar proportion, saturated fat proportion
- **Final model features** (5+):
  - Numeric: Calories, sugar prop, sat fat prop, protein prop
  - Text: CountVectorized ingredients (500+ features)

### Step 3: Data Filtering & Splitting
- Filters to recipes tagged as ONLY American OR ONLY European
- Removes rows with missing values
- 80/20 train/test split (stratified)
- Random state: 52 (for reproducibility)

**Resulting dataset**:
- Total: 17,214 recipes
- Train: 13,771 samples
- Test: 3,443 samples
- Class balance: 51% USA, 49% EUR

### Step 4: Model Training
**Baseline Model**:
- Algorithm: Random Forest (100 trees)
- Features: 3 (macronutrients)
- Result: 54.14% test accuracy

**Final Model**:
- Algorithm: Random Forest with optimized hyperparameters
- Criterion: Entropy (better for balanced classes)
- Max depth: 100 (balances capacity and regularization)
- Features: Ingredients + 4 numeric features
- Result: 76.01% test accuracy

### Step 5: Evaluation
Computes comprehensive metrics:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Discrimination ability
- **Cross-Validation**: 5-fold stratified CV (mean ± std)
- **Confidence Intervals**: 95% CI using normal approximation
- **Confusion Matrices**: For interpretation

### Step 6: Visualization
Generates 3 publication-quality figures:

**Figure 1: ROC Curves**
- Compares baseline (AUC=0.5596) vs final model (AUC=0.8376)
- Shows improvement in discrimination

**Figure 2: Learning Curves**
- Baseline: High bias (underfitting)
- Final: High variance (controlled by max_depth regularization)
- Demonstrates regularization effectiveness

**Figure 3: Feature Importance**
- Top 20 most important features
- Shows nutritional vs ingredient contribution
- Identifies key discriminative signals

---

## 🔧 Configuration

All key parameters are defined at the top of the script:

```python
# Random seed (must be consistent)
np.random.seed(52)

# Data paths (update if needed)
base_path = Path('./Archive')  # Contains RAW_recipes.csv, interactions.csv

# Model parameters
random_state = 52
test_size = 0.2

# Final model hyperparameters
criterion = 'entropy'
max_depth = 100
```

**To change parameters**:
1. Edit the values in the script
2. Run: `python3 generate_metrics.py`
3. Check outputs in `../data/` and `../assets/papers/`

---

## 📈 Reproducibility

**Guaranteed reproducible** due to:
- Fixed random seed (52) throughout pipeline
- Stratified train/test split (preserves class balance)
- Deterministic cross-validation folds
- No randomization in model after seed set

**To verify reproducibility**:
```bash
# Run twice and compare outputs
python3 generate_metrics.py
cp ../data/metrics_summary.json metrics_v1.json

python3 generate_metrics.py
diff metrics_v1.json ../data/metrics_summary.json

# Should be identical (no diff output)
```

---

## 💻 System Requirements

**Python Version**: 3.7+

**Required Packages**:
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
```

**Install**:
```bash
pip install scikit-learn pandas numpy matplotlib
```

**Data**: Requires data files in `../Archive/`:
- `RAW_recipes.csv` (83,782 recipes)
- `interactions.csv` (731,927 reviews)

---

## 📊 Output Interpretation

### metrics_summary.json
```json
{
  "final_metrics": {
    "test_accuracy": 0.7601,              # 76.01% on test set
    "cv_mean": 0.7594,                    # 75.94% average across 5 folds
    "cv_std": 0.0054,                     # Very consistent (±0.54%)
    "accuracy_ci": [0.7458, 0.7744],      # 95% CI
    "roc_auc": 0.8376,                    # Strong discrimination
    "top_20_features": [...]              # Feature rankings
  }
}
```

### ROC Curves
- Higher curve = better model
- Area under curve (AUC) = probability model ranks random positive higher
- Final model AUC 0.8376 indicates strong discrimination

### Learning Curves
- Training curve high, validation lower = overfitting (but controlled)
- Curves converging = good regularization balance
- Space between curves = room to improve with more data

### Feature Importance
- Sugar proportion: Most important single feature
- Ingredients: Collective strongest signal (76% of importance)
- Olive oil, salt, garlic: European signature
- Butter, chili: American signature

---

## 🐛 Troubleshooting

**Error: "FileNotFoundError: RAW_recipes.csv"**
- Ensure data files are in `../Archive/` directory
- Check path in script matches your setup

**Error: "ModuleNotFoundError: sklearn"**
- Install: `pip install scikit-learn`

**Warning: "Matplotlib is building font cache"**
- Normal on first run, takes a minute
- Subsequent runs will be faster

**Output files not created**
- Check script runs to completion
- Verify write permissions in `../data/` and `../assets/papers/`
- Ensure directories exist (mkdir -p if needed)

---

## 📝 Extending the Script

To add more analysis:

```python
# Example: Add additional metric
from sklearn.metrics import roc_auc_score

# After training final_model:
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
other_metric = roc_auc_score(y_test, y_pred_proba)
print(f"Additional metric: {other_metric:.4f}")
```

---

## 🔗 Related Resources

- **Technical Paper**: `../papers/ml-classification.md`
- **Results Data**: `../data/`
- **Visualizations**: `../assets/papers/`
- **Original Notebook**: `../papers/code-notebook.md`

---

## 📞 Usage Tips

**Run silently (no output)**:
```bash
python3 generate_metrics.py > /dev/null 2>&1
```

**Run with verbose logging**:
```bash
python3 -u generate_metrics.py  # Force unbuffered output
```

**Integrate into workflow**:
```bash
# Regenerate and immediately check results
python3 generate_metrics.py && cat ../data/results_table.csv
```

**Schedule periodic updates** (e.g., via cron):
```bash
# Update metrics daily
0 6 * * * cd /path/to/scripts && python3 generate_metrics.py
```

---

## ✨ Version History

- **v1.0** (Mar 28, 2026): Initial reproducible ML pipeline
  - Comprehensive data preprocessing
  - Baseline and final model training
  - 5-fold cross-validation
  - 95% confidence intervals
  - 3 publication-quality visualizations

---

*Last updated: Mar 28, 2026*
*Compatibility: Python 3.7+, scikit-learn 1.0+*
