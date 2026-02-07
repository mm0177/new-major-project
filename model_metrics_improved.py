import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import shap

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def train_improved_model_with_smote():
    """
    Train an improved model with SMOTE to handle class imbalance
    """
    print("="*70)
    print("TRAINING IMPROVED MODEL WITH CLASS BALANCING")
    print("="*70 + "\n")
    
    # Load dataset
    df = pd.read_csv('final_fused_dataset.csv')
    
    features = ['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']
    X = df[features]
    y = df['risk_label']
    
    print(f"Original Dataset Class Distribution:")
    print(f"  No Risk (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.2f}%)")
    print(f"  Risk (1):    {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.2f}%)")
    
    # CRITICAL FIX: Use stratified split to ensure both classes in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nAfter Stratified Split:")
    print(f"  Train - Risk cases: {(y_train == 1).sum()} / {len(y_train)}")
    print(f"  Test  - Risk cases: {(y_test == 1).sum()} / {len(y_test)}")
    
    # Apply SMOTE to balance training data
    print("\n⚙️  Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE Balancing:")
    print(f"  No Risk (0): {(y_train_balanced == 0).sum()}")
    print(f"  Risk (1):    {(y_train_balanced == 1).sum()}")
    
    # Train improved model with class weights
    print("\n⚙️  Training XGBoost with balanced data...")
    model_improved = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Class weight
        random_state=42
    )
    model_improved.fit(X_train_balanced, y_train_balanced)
    
    # Save improved model
    joblib.dump(model_improved, 'risk_engine_improved.pkl')
    print("\n✅ Improved model saved as 'risk_engine_improved.pkl'")
    
    return model_improved, X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced


def generate_improved_metrics():
    """
    Generate comprehensive metrics with fixes for class imbalance issues
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL METRICS - IMPROVED VERSION")
    print("="*70 + "\n")
    
    # Train improved model
    model, X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced = train_improved_model_with_smote()
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ========================================================================
    # SECTION 1: IMPROVED CLASSIFICATION METRICS
    # ========================================================================
    print("\n" + "="*70)
    print("1. IMPROVED CLASSIFICATION METRICS")
    print("="*70)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nImproved Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n" + "-"*70)
    print("Detailed Classification Report:")
    print("-"*70)
    print(classification_report(y_test, y_pred, target_names=['No Risk', 'Risk']))
    
    # ========================================================================
    # SECTION 2: IMPROVED CONFUSION MATRIX
    # ========================================================================
    print("\n" + "="*70)
    print("2. IMPROVED CONFUSION MATRIX")
    print("="*70)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nDetailed Breakdown:")
    print(f"  True Negatives (No Risk → No Risk):  {cm[0, 0]}")
    print(f"  False Positives (No Risk → Risk):    {cm[0, 1]}")
    print(f"  False Negatives (Risk → No Risk):    {cm[1, 0]} ⚠️  MISSED RISKS!")
    print(f"  True Positives (Risk → Risk):        {cm[1, 1]} ✅ CAUGHT RISKS!")
    
    # Visualize Improved Confusion Matrix with annotations
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['No Risk', 'Risk'],
                yticklabels=['No Risk', 'Risk'],
                cbar_kws={'label': 'Count'})
    plt.title('Improved Confusion Matrix - Supply Chain Risk Prediction', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm.sum() * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig('research_metrics/confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    print("\n✅ Improved confusion matrix saved")
    plt.close()
    
    # ========================================================================
    # SECTION 3: IMPROVED ROC CURVE
    # ========================================================================
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Improved Model (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Improved Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('research_metrics/roc_curve_improved.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Improved ROC-AUC: {roc_auc:.4f}")
    plt.close()
    
    # ========================================================================
    # SECTION 4: FIXED CROSS-VALIDATION with STRATIFIED FOLDS
    # ========================================================================
    print("\n" + "="*70)
    print("4. FIXED CROSS-VALIDATION (Stratified)")
    print("="*70)
    
    # Load original data for CV
    df = pd.read_csv('final_fused_dataset.csv')
    features = ['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']
    X_full = df[features]
    y_full = df['risk_label']
    
    # Use StratifiedKFold to ensure each fold has both classes
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    cv_results = cross_validate(
        model, X_full, y_full, 
        cv=skf, 
        scoring=scoring,
        return_train_score=False
    )
    
    print("\n5-Fold Stratified Cross-Validation Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        scores = cv_results[f'test_{metric}']
        print(f"  {metric.capitalize():10s}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Visualize CV scores with proper bars
    cv_df = pd.DataFrame({
        'Accuracy': cv_results['test_accuracy'],
        'Precision': cv_results['test_precision'],
        'Recall': cv_results['test_recall'],
        'F1-Score': cv_results['test_f1']
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    cv_df.boxplot(ax=ax1)
    ax1.set_title('Stratified 5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Bar plot with error bars
    means = cv_df.mean()
    stds = cv_df.std()
    x_pos = np.arange(len(means))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(means.index, rotation=15)
    ax2.set_ylabel('Mean Score', fontsize=12)
    ax2.set_title('Mean CV Scores with Std Dev', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax2.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('research_metrics/cross_validation_improved.png', dpi=300, bbox_inches='tight')
    print("\n✅ Fixed cross-validation plot saved")
    plt.close()
    
    # ========================================================================
    # SECTION 5: CLASS IMBALANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("5. CLASS IMBALANCE ANALYSIS")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original distribution
    original_counts = y_full.value_counts()
    axes[0].bar(['No Risk', 'Risk'], original_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Original Dataset', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=10)
    for i, v in enumerate(original_counts.values):
        axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    # Training set after SMOTE
    train_balanced_counts = pd.Series(y_train_balanced).value_counts()
    axes[1].bar(['No Risk', 'Risk'], train_balanced_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Training Set (After SMOTE)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=10)
    for i, v in enumerate(train_balanced_counts.values):
        axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    # Test set
    test_counts = y_test.value_counts()
    axes[2].bar(['No Risk', 'Risk'], test_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[2].set_title('Test Set (Stratified)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Count', fontsize=10)
    for i, v in enumerate(test_counts.values):
        axes[2].text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    plt.suptitle('Class Distribution Across Datasets', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('research_metrics/class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Class imbalance analysis saved")
    plt.close()
    
    # ========================================================================
    # EXPORT IMPROVED METRICS
    # ========================================================================
    metrics_summary = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Improved_Model': [accuracy, precision, recall, f1, roc_auc]
    }
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv('research_metrics/model_metrics_improved.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ IMPROVED EVALUATION COMPLETE!")
    print("="*70)
    print("\n📊 New Files Generated:")
    print("  1. confusion_matrix_improved.png (with detailed annotations)")
    print("  2. cross_validation_improved.png (FIXED - all bars visible!)")
    print("  3. roc_curve_improved.png")
    print("  4. class_imbalance_analysis.png (shows SMOTE effect)")
    print("  5. model_metrics_improved.csv")
    print("  6. risk_engine_improved.pkl (improved model)")
    print("\n")


if __name__ == "__main__":
    generate_improved_metrics()
