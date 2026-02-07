# Baseline Model - Evaluation Results

## Overview
This folder contains the evaluation metrics for the **baseline XGBoost model** trained on the original imbalanced dataset without any class balancing techniques.

---

## Model Configuration

### Training Approach
- **Algorithm**: XGBoost Classifier
- **Train-Test Split**: 80-20 (non-stratified)
- **Class Balancing**: None (trained on imbalanced data)
- **Hyperparameters**:
  - n_estimators: 100
  - learning_rate: 0.05
  - max_depth: 6
  - random_state: 42

### Dataset Characteristics
- **Total Samples**: 164
- **Training Samples**: 131 (80%)
- **Testing Samples**: 33 (20%)
- **Class Distribution**:
  - No Risk (0): 123 samples (75%)
  - Risk (1): 41 samples (25%) ⚠️ **Imbalanced**

---

## Performance Metrics

### Classification Results

| Metric | Value | Percentage |
|--------|-------|------------|
| **Accuracy** | 0.8780 | **87.80%** ✅ |
| **Precision** | 0.6000 | 60.00% |
| **Recall** | 0.5000 | 50.00% ⚠️ |
| **F1-Score** | 0.5455 | 54.55% |
| **AUC-ROC** | 0.7095 | 70.95% |
| **Average Precision** | 0.5646 | 56.46% |

### Cross-Validation (5-Fold)
- **Mean Accuracy**: 79.63% ± 11.90%
- **Note**: Precision, Recall, and F1-Score metrics had issues due to non-stratified CV

---

## Confusion Matrix

```
              Predicted
              No Risk  Risk
Actual  
No Risk    27        2      (93% correct)
Risk        3        3      (50% caught) ⚠️
```

### Breakdown:
- **True Negatives**: 27 (correctly predicted no risk)
- **False Positives**: 2 (false alarms)
- **False Negatives**: 3 ⚠️ **MISSED RISKS** (actual risks not detected)
- **True Positives**: 3 ✅ (correctly caught risks)

**Critical Issue**: The model **missed 50% of actual supply chain risks** (3 out of 6 risk cases)!

---

## Key Findings

### Strengths ✅
1. **High Overall Accuracy** (87.80%)
2. **Low False Positive Rate** (only 2 false alarms)
3. **Good Performance on Majority Class** (93% correct for "No Risk")

### Weaknesses ⚠️
1. **Poor Recall** (50%) - Misses half of actual risks
2. **Class Imbalance Bias** - Model favors predicting "No Risk"
3. **Low True Positives** - Only caught 3 out of 6 risk events
4. **Cross-Validation Issues** - Unstratified folds caused metric calculation problems

---

## Feature Importance

Based on XGBoost feature importance scores:

1. **Sentiment Score** (BERT news sentiment) - Most important
2. **News Volume** (Article count)
3. **Precipitation (PRCP)** (Weather)
4. **Average Temperature (TAVG)** (Weather)
5. **Wind Speed (WSPD)** (Weather)

---

## Files in This Folder

### Visualizations (8 PNG files)
1. **confusion_matrix.png** - Confusion matrix heatmap
2. **roc_curve.png** - ROC curve with AUC = 70.95%
3. **precision_recall_curve.png** - Precision-Recall curve
4. **cross_validation_scores.png** - CV scores (⚠️ missing bars for precision/recall/F1)
5. **feature_importance.png** - Feature importance ranking
6. **shap_summary_plot.png** - SHAP values distribution
7. **shap_bar_plot.png** - Mean absolute SHAP values
8. **class_distribution.png** - Dataset class balance visualization

### Data Files (2 CSV files)
9. **model_metrics_summary.csv** - All metrics in tabular format
10. **detailed_predictions.csv** - Individual prediction results with probabilities

---

## Interpretation & Analysis

### Why High Accuracy But Low Recall?

The high accuracy (87.80%) is **misleading** due to class imbalance:

- Since 75% of data is "No Risk", the model learned to predict "No Risk" most of the time
- This strategy gives high accuracy but **fails to catch actual risks**
- The model is **conservative** - it avoids false alarms but misses real disruptions

### The Real-World Impact

In a supply chain context:
- ✅ **True Positive (Caught Risk)**: Company prepares → Avoids disruption
- ❌ **False Negative (Missed Risk)**: No preparation → Disruption causes losses
- ⚠️ **False Positive (False Alarm)**: Unnecessary precaution → Minor cost
- ✅ **True Negative (Correct No-Risk)**: Business as usual

**Missing risks (False Negatives) is the worst outcome!** This baseline model has 3 false negatives.

---

## Recommendations

### For Research Paper
1. **Present this as the baseline** to compare against improved approaches
2. **Highlight the accuracy paradox** - high accuracy doesn't mean good performance
3. **Discuss class imbalance** as a key challenge in the domain
4. **Explain why recall matters more than accuracy** for risk prediction

### For Model Improvement
See the `improved_model_smote/` folder for the enhanced version that addresses these issues!

---

## Technical Details

### Model Training Code
```python
model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=6
)
model.fit(X_train, y_train)
```

### Evaluation Methodology
- Scikit-learn metrics (accuracy, precision, recall, F1)
- ROC-AUC for probability-based evaluation
- SHAP for explainability
- 5-fold cross-validation for robustness testing

---

## Conclusion

This baseline model demonstrates **good overall accuracy but poor risk detection capability**. The severe class imbalance (75-25 split) causes the model to be biased toward the majority class, resulting in **missed risks**.

**Next Step**: Review the `improved_model_smote/` folder to see how SMOTE and stratified sampling address these issues.

---

**Model File**: `risk_engine.pkl` (located in parent directory)  
**Evaluation Script**: `model_metrics_evaluation.py` (located in parent directory)  
**Generated**: 2026-02-07
