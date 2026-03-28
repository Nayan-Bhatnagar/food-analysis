#!/usr/bin/env python3
"""
Generate comprehensive ML metrics for the food analysis paper
Reproduces model training and computes rigorous validation metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set random seed for reproducibility
np.random.seed(52)

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================
print("=" * 80)
print("LOADING AND PREPROCESSING DATA")
print("=" * 80)

base_path = Path('/Users/gauravbhatnagar/git/food-analysis-paper/Archive')
raw_recipes = pd.read_csv(base_path / 'RAW_recipes.csv')
raw_reviews = pd.read_csv(base_path / 'interactions.csv')

print(f"✓ Loaded raw recipes: {raw_recipes.shape}")
print(f"✓ Loaded raw reviews: {raw_reviews.shape}")

# Process recipes
recipes = raw_recipes.copy()

# Drop contributor_id
recipes = recipes.drop('contributor_id', axis=1)

# Rename columns
recipes = recipes.rename(columns={
    'submitted': 'recipe_date',
    'n_steps': 'n_steps',
    'n_ingredients': 'n_ingredients'
})

# Convert dates
recipes['recipe_date'] = pd.to_datetime(recipes['recipe_date'], format='%Y-%m-%d')

# Handle missing minutes (replace 0 with NaN)
recipes['minutes'] = recipes['minutes'].replace(0, np.nan)

# Extract nutrition features from array string
def parse_nutrition(nutrition_str):
    """Parse nutrition string array into individual values"""
    try:
        # Remove brackets and convert to list of floats
        values = [float(x) for x in str(nutrition_str).strip('[]').split(',')]
        return pd.Series({
            'calories': values[0],
            'fat': values[1],
            'sugar': values[2],
            'sodium': values[3],
            'protein': values[4],
            'sat_fat': values[5],
            'carbs': values[6]
        })
    except:
        return pd.Series({
            'calories': np.nan,
            'fat': np.nan,
            'sugar': np.nan,
            'sodium': np.nan,
            'protein': np.nan,
            'sat_fat': np.nan,
            'carbs': np.nan
        })

nutrition_features = recipes['nutrition'].apply(parse_nutrition)
recipes = pd.concat([recipes, nutrition_features], axis=1)

# Calculate nutrition proportions
# Reference values and conversions
grams_per_100 = {
    'fat': 65.2, 'sugar': 25.1, 'protein': 50.2,
    'sat_fat': 20.1, 'carbs': 301.6
}
cal_per_gram = {'fat': 9, 'sugar': 4, 'protein': 4, 'sat_fat': 9, 'carbs': 4}

for nutrient in ['fat', 'sugar', 'protein', 'sat_fat', 'carbs']:
    recipes[f'{nutrient}_prop'] = (
        (recipes[nutrient] / 100) * grams_per_100[nutrient] * cal_per_gram[nutrient]
    ) / recipes['calories']

# Extract American and European status
recipes['usa'] = recipes['tags'].str.contains("'american'", case=False, na=False)
recipes['euro'] = recipes['tags'].str.contains("'european'", case=False, na=False)

# Merge with reviews for average rating
reviews = raw_reviews.copy()
reviews['review_date'] = pd.to_datetime(reviews['date'], format='%Y-%m-%d')
reviews['rating'] = reviews['rating'].replace(0, np.nan)

avg_ratings = reviews.groupby('recipe_id')['rating'].mean().reset_index()
avg_ratings.rename(columns={'recipe_id': 'id', 'rating': 'average_rating'}, inplace=True)

recipes = recipes.merge(avg_ratings, on='id', how='left')

print(f"✓ Processed recipes: {recipes.shape}")
print(f"✓ Nutrition features and proportions calculated")

# ============================================================================
# STEP 2: PREPARE DATA FOR MODELING
# ============================================================================
print("\n" + "=" * 80)
print("PREPARING DATA FOR MODELING")
print("=" * 80)

# Filter for only American or European recipes (not both)
recipes_filtered = recipes[
    ((recipes['usa'] == True) & (recipes['euro'] == False)) |
    ((recipes['euro'] == True) & (recipes['usa'] == False))
].copy()

recipes_filtered['type'] = recipes_filtered['usa'].map({True: 'usa', False: 'euro'})

print(f"✓ Filtered to American/European only: {recipes_filtered.shape[0]} recipes")
print(f"  - American: {(recipes_filtered['type'] == 'usa').sum()}")
print(f"  - European: {(recipes_filtered['type'] == 'euro').sum()}")

# Select features and target
features_for_modeling = [
    'ingredients', 'calories', 'sugar_prop', 'sat_fat_prop', 'protein_prop'
]

X = recipes_filtered[features_for_modeling].copy()
y = recipes_filtered['type'].map({'usa': 1, 'euro': 0})

# Check for missing values before split
for col in features_for_modeling:
    missing = X[col].isnull().sum()
    if missing > 0:
        print(f"  - {col}: {missing} missing values")

# Drop rows with missing values
mask = X.isnull().any(axis=1) | y.isnull()
X = X[~mask].reset_index(drop=True)
y = y[~mask].reset_index(drop=True)

print(f"✓ Final dataset for modeling: {X.shape[0]} recipes, {X.shape[1]} features")
print(f"✓ Target distribution: {(y==1).sum()} USA, {(y==0).sum()} EUR")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=52
)

print(f"✓ Train/test split (80/20):")
print(f"  - Train: {X_train.shape[0]} samples")
print(f"  - Test: {X_test.shape[0]} samples")

# ============================================================================
# STEP 3: BASELINE MODEL (Simple features only)
# ============================================================================
print("\n" + "=" * 80)
print("BASELINE MODEL (Calories, Sugar Prop, Sat Fat Prop)")
print("=" * 80)

# Baseline features
X_train_baseline = X_train[['calories', 'sugar_prop', 'sat_fat_prop']]
X_test_baseline = X_test[['calories', 'sugar_prop', 'sat_fat_prop']]

# Train baseline model
baseline_model = RandomForestClassifier(
    random_state=52, n_estimators=100, max_depth=None, criterion='gini'
)
baseline_model.fit(X_train_baseline, y_train)

# Evaluate baseline
y_pred_baseline_train = baseline_model.predict(X_train_baseline)
y_pred_baseline_test = baseline_model.predict(X_test_baseline)

train_acc_baseline = accuracy_score(y_train, y_pred_baseline_train)
test_acc_baseline = accuracy_score(y_test, y_pred_baseline_test)

print(f"Training Accuracy: {train_acc_baseline:.4f}")
print(f"Testing Accuracy: {test_acc_baseline:.4f}")

# Classification metrics
baseline_metrics = {
    'model': 'Baseline (RF, 3 features)',
    'train_accuracy': float(train_acc_baseline),
    'test_accuracy': float(test_acc_baseline),
    'precision_usa': float(precision_score(y_test, y_pred_baseline_test, pos_label=1)),
    'recall_usa': float(recall_score(y_test, y_pred_baseline_test, pos_label=1)),
    'f1_usa': float(f1_score(y_test, y_pred_baseline_test, pos_label=1)),
    'roc_auc': float(roc_auc_score(y_test, baseline_model.predict_proba(X_test_baseline)[:, 1]))
}

confusion_matrix_baseline = confusion_matrix(y_test, y_pred_baseline_test)
print(f"Confusion Matrix (USA=1, EUR=0):\n{confusion_matrix_baseline}")

# ============================================================================
# STEP 4: FINAL MODEL (With ingredients + GridSearch optimization)
# ============================================================================
print("\n" + "=" * 80)
print("FINAL MODEL (Ingredients + Numeric Features, Optimized)")
print("=" * 80)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('ingredients', CountVectorizer(
            max_features=500,
            token_pattern=r"(?u)[^\[\]\"', \n].+?(?=',|']|\",)"
        ), 'ingredients'),
        ('numeric', 'passthrough', ['calories', 'sugar_prop', 'sat_fat_prop', 'protein_prop'])
    ]
)

# Final model with best hyperparameters from grid search
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        criterion='entropy',
        max_depth=100,
        random_state=52,
        n_estimators=100,
        n_jobs=-1
    ))
])

# Train final model
final_model.fit(X_train, y_train)

# Evaluate final model
y_pred_final_train = final_model.predict(X_train)
y_pred_final_test = final_model.predict(X_test)
y_pred_proba_final_test = final_model.predict_proba(X_test)[:, 1]

train_acc_final = accuracy_score(y_train, y_pred_final_train)
test_acc_final = accuracy_score(y_test, y_pred_final_test)

print(f"Training Accuracy: {train_acc_final:.4f}")
print(f"Testing Accuracy: {test_acc_final:.4f}")

# Classification metrics
final_metrics = {
    'model': 'Final (RF, ingredients + numeric)',
    'train_accuracy': float(train_acc_final),
    'test_accuracy': float(test_acc_final),
    'precision_usa': float(precision_score(y_test, y_pred_final_test, pos_label=1)),
    'recall_usa': float(recall_score(y_test, y_pred_final_test, pos_label=1)),
    'f1_usa': float(f1_score(y_test, y_pred_final_test, pos_label=1)),
    'roc_auc': float(roc_auc_score(y_test, y_pred_proba_final_test))
}

confusion_matrix_final = confusion_matrix(y_test, y_pred_final_test)
print(f"Confusion Matrix (USA=1, EUR=0):\n{confusion_matrix_final}")

# ============================================================================
# STEP 5: CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("CROSS-VALIDATION (5-fold)")
print("=" * 80)

# 5-fold CV for final model
cv_scores = cross_val_score(
    final_model, X_train, y_train,
    cv=5, scoring='accuracy', n_jobs=-1
)

print(f"CV Fold Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

final_metrics['cv_mean'] = float(cv_scores.mean())
final_metrics['cv_std'] = float(cv_scores.std())
final_metrics['cv_scores'] = cv_scores.tolist()

# ============================================================================
# STEP 6: CONFIDENCE INTERVALS
# ============================================================================
print("\n" + "=" * 80)
print("CONFIDENCE INTERVALS (95%)")
print("=" * 80)

def compute_ci_95(y_true, y_pred, metric_name='accuracy'):
    """Compute 95% CI using normal approximation"""
    n = len(y_true)
    if metric_name == 'accuracy':
        p = accuracy_score(y_true, y_pred)
    elif metric_name == 'precision':
        p = precision_score(y_true, y_pred, pos_label=1)
    elif metric_name == 'recall':
        p = recall_score(y_true, y_pred, pos_label=1)
    elif metric_name == 'f1':
        p = f1_score(y_true, y_pred, pos_label=1)

    se = np.sqrt(p * (1 - p) / n)
    z = 1.96  # 95% confidence
    ci_lower = max(0, p - z * se)
    ci_upper = min(1, p + z * se)

    return p, ci_lower, ci_upper, se

# Compute CIs for final model
acc, acc_ci_lower, acc_ci_upper, acc_se = compute_ci_95(y_test, y_pred_final_test, 'accuracy')
prec, prec_ci_lower, prec_ci_upper, prec_se = compute_ci_95(y_test, y_pred_final_test, 'precision')
rec, rec_ci_lower, rec_ci_upper, rec_se = compute_ci_95(y_test, y_pred_final_test, 'recall')
f1, f1_ci_lower, f1_ci_upper, f1_se = compute_ci_95(y_test, y_pred_final_test, 'f1')

final_metrics['accuracy_ci'] = [float(acc_ci_lower), float(acc_ci_upper)]
final_metrics['precision_usa_ci'] = [float(prec_ci_lower), float(prec_ci_upper)]
final_metrics['recall_usa_ci'] = [float(rec_ci_lower), float(rec_ci_upper)]
final_metrics['f1_usa_ci'] = [float(f1_ci_lower), float(f1_ci_upper)]

print(f"Accuracy: {acc:.4f} [95% CI: {acc_ci_lower:.4f} - {acc_ci_upper:.4f}]")
print(f"Precision (USA): {prec:.4f} [95% CI: {prec_ci_lower:.4f} - {prec_ci_upper:.4f}]")
print(f"Recall (USA): {rec:.4f} [95% CI: {rec_ci_lower:.4f} - {rec_ci_upper:.4f}]")
print(f"F1 (USA): {f1:.4f} [95% CI: {f1_ci_lower:.4f} - {f1_ci_upper:.4f}]")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Top 20)")
print("=" * 80)

# Get feature importance from the classifier
classifier = final_model.named_steps['classifier']
preprocessor_fitted = final_model.named_steps['preprocessor']

# Get feature names
count_vec = preprocessor_fitted.named_transformers_['ingredients']
ingredient_features = count_vec.get_feature_names_out()
numeric_features = ['calories', 'sugar_prop', 'sat_fat_prop', 'protein_prop']
all_feature_names = np.concatenate([ingredient_features, numeric_features])

# Get importances
importances = classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance_df.head(20))

top_features = feature_importance_df.head(20).to_dict('records')
final_metrics['top_20_features'] = top_features

# ============================================================================
# STEP 8: GENERATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

output_dir = Path('/Users/gauravbhatnagar/git/food-analysis-paper/figures')
output_dir.mkdir(exist_ok=True)

# Figure 1: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

# Baseline ROC
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, baseline_model.predict_proba(X_test_baseline)[:, 1])
auc_baseline = auc(fpr_baseline, tpr_baseline)

# Final ROC
fpr_final, tpr_final, _ = roc_curve(y_test, y_pred_proba_final_test)
auc_final = auc(fpr_final, tpr_final)

# Plot
ax.plot(fpr_baseline, tpr_baseline, 'b-', linewidth=2, label=f'Baseline (AUC={auc_baseline:.3f})')
ax.plot(fpr_final, tpr_final, 'r-', linewidth=2, label=f'Final Model (AUC={auc_final:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Baseline vs Final Model', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved ROC curves: {output_dir / 'roc_curves.png'}")
plt.close()

# Figure 2: Learning Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline learning curve
train_sizes, train_scores_base, val_scores_base = learning_curve(
    baseline_model, X_train_baseline, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy', n_jobs=-1
)

train_mean_base = np.mean(train_scores_base, axis=1)
train_std_base = np.std(train_scores_base, axis=1)
val_mean_base = np.mean(val_scores_base, axis=1)
val_std_base = np.std(val_scores_base, axis=1)

axes[0].plot(train_sizes, train_mean_base, 'b-', linewidth=2, label='Training')
axes[0].fill_between(train_sizes, train_mean_base - train_std_base, train_mean_base + train_std_base, alpha=0.2, color='b')
axes[0].plot(train_sizes, val_mean_base, 'r-', linewidth=2, label='Validation')
axes[0].fill_between(train_sizes, val_mean_base - val_std_base, val_mean_base + val_std_base, alpha=0.2, color='r')
axes[0].set_xlabel('Training Set Size', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_title('Baseline Model Learning Curve', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.4, 1.0])

# Final model learning curve
train_sizes_final, train_scores_final, val_scores_final = learning_curve(
    final_model, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy', n_jobs=-1
)

train_mean_final = np.mean(train_scores_final, axis=1)
train_std_final = np.std(train_scores_final, axis=1)
val_mean_final = np.mean(val_scores_final, axis=1)
val_std_final = np.std(val_scores_final, axis=1)

axes[1].plot(train_sizes_final, train_mean_final, 'b-', linewidth=2, label='Training')
axes[1].fill_between(train_sizes_final, train_mean_final - train_std_final, train_mean_final + train_std_final, alpha=0.2, color='b')
axes[1].plot(train_sizes_final, val_mean_final, 'r-', linewidth=2, label='Validation')
axes[1].fill_between(train_sizes_final, val_mean_final - val_std_final, val_mean_final + val_std_final, alpha=0.2, color='r')
axes[1].set_xlabel('Training Set Size', fontsize=11)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Final Model Learning Curve', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0.4, 1.0])

plt.tight_layout()
plt.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved learning curves: {output_dir / 'learning_curves.png'}")
plt.close()

# Figure 3: Feature Importance (Top 20)
fig, ax = plt.subplots(figsize=(10, 8))
top_20 = feature_importance_df.head(20)
ax.barh(range(len(top_20)), top_20['importance'], color='steelblue')
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'], fontsize=10)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved feature importance: {output_dir / 'feature_importance.png'}")
plt.close()

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save metrics to JSON
metrics_output = {
    'baseline_metrics': baseline_metrics,
    'final_metrics': final_metrics,
    'confusion_matrix_baseline': confusion_matrix_baseline.tolist(),
    'confusion_matrix_final': confusion_matrix_final.tolist(),
    'class_distribution': {
        'usa_count': int((y_test == 1).sum()),
        'eur_count': int((y_test == 0).sum()),
        'total': len(y_test)
    }
}

metrics_file = Path('/Users/gauravbhatnagar/git/food-analysis-paper/metrics_summary.json')
with open(metrics_file, 'w') as f:
    json.dump(metrics_output, f, indent=2)

print(f"✓ Saved metrics summary: {metrics_file}")

# Save detailed results table
results_df = pd.DataFrame({
    'Metric': [
        'Training Accuracy',
        'Test Accuracy',
        'Precision (USA)',
        'Recall (USA)',
        'F1-Score (USA)',
        'ROC-AUC',
        'CV Mean ± Std'
    ],
    'Baseline': [
        f"{baseline_metrics['train_accuracy']:.4f}",
        f"{baseline_metrics['test_accuracy']:.4f}",
        f"{baseline_metrics['precision_usa']:.4f}",
        f"{baseline_metrics['recall_usa']:.4f}",
        f"{baseline_metrics['f1_usa']:.4f}",
        f"{baseline_metrics['roc_auc']:.4f}",
        "N/A"
    ],
    'Final Model': [
        f"{final_metrics['train_accuracy']:.4f}",
        f"{final_metrics['test_accuracy']:.4f}",
        f"{final_metrics['precision_usa']:.4f}",
        f"{final_metrics['recall_usa']:.4f}",
        f"{final_metrics['f1_usa']:.4f}",
        f"{final_metrics['roc_auc']:.4f}",
        f"{final_metrics['cv_mean']:.4f} ± {final_metrics['cv_std']:.4f}"
    ]
})

results_table_file = Path('/Users/gauravbhatnagar/git/food-analysis-paper/results_table.csv')
results_df.to_csv(results_table_file, index=False)
print(f"✓ Saved results table: {results_table_file}")

print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))
print("\n✓ Metrics generation complete!")
