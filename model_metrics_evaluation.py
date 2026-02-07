import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
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
import shap

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def generate_comprehensive_metrics():
    """
    Generate comprehensive metrics for the XGBoost Supply Chain Risk Model
    Suitable for research paper publication
    """
    print("="*70)
    print("COMPREHENSIVE MODEL METRICS EVALUATION")
    print("XGBoost-based Supply Chain Risk Prediction Model")
    print("="*70 + "\n")
    
    # 1. Load the fused dataset and trained model
    try:
        df = pd.read_csv('final_fused_dataset.csv')
        model = joblib.load('risk_engine.pkl')
        print("✅ Successfully loaded dataset and trained model\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run 'fusion_xb_boost.py' first to generate required files.")
        return
    
    # 2. Prepare features and target
    features = ['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']
    X = df[features]
    y = df['risk_label']
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ========================================================================
    # SECTION 1: BASIC CLASSIFICATION METRICS
    # ========================================================================
    print("\n" + "="*70)
    print("1. CLASSIFICATION METRICS")
    print("="*70)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print("\n" + "-"*70)
    print("Detailed Classification Report:")
    print("-"*70)
    print(classification_report(y_test, y_pred, target_names=['No Risk', 'Risk']))
    
    # ========================================================================
    # SECTION 2: CONFUSION MATRIX
    # ========================================================================
    print("\n" + "="*70)
    print("2. CONFUSION MATRIX")
    print("="*70)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Risk', 'Risk'],
                yticklabels=['No Risk', 'Risk'])
    plt.title('Confusion Matrix - Supply Chain Risk Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
    # ========================================================================
    # SECTION 3: ROC CURVE AND AUC
    # ========================================================================
    print("\n" + "="*70)
    print("3. ROC CURVE AND AUC SCORE")
    print("="*70)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nAUC-ROC Score: {roc_auc:.4f}")
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Supply Chain Risk Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✅ ROC curve saved as 'roc_curve.png'")
    plt.close()
    
    # ========================================================================
    # SECTION 4: PRECISION-RECALL CURVE
    # ========================================================================
    print("\n" + "="*70)
    print("4. PRECISION-RECALL CURVE")
    print("="*70)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    print(f"\nAverage Precision Score: {avg_precision:.4f}")
    
    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print("✅ Precision-Recall curve saved as 'precision_recall_curve.png'")
    plt.close()
    
    # ========================================================================
    # SECTION 5: CROSS-VALIDATION SCORES
    # ========================================================================
    print("\n" + "="*70)
    print("5. CROSS-VALIDATION ANALYSIS")
    print("="*70)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision', )
    cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"  Accuracy:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"  Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"  F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    
    # Visualize Cross-Validation Scores
    cv_results = pd.DataFrame({
        'Accuracy': cv_scores,
        'Precision': cv_precision,
        'Recall': cv_recall,
        'F1-Score': cv_f1
    })
    
    plt.figure(figsize=(10, 6))
    cv_results.boxplot()
    plt.title('5-Fold Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cross_validation_scores.png', dpi=300, bbox_inches='tight')
    print("✅ Cross-validation plot saved as 'cross_validation_scores.png'")
    plt.close()
    
    # ========================================================================
    # SECTION 6: FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "="*70)
    print("6. FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    feature_importance = model.feature_importances_
    feature_names = features
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance Ranking:")
    for idx, row in importance_df.iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")
    
    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(features))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance - XGBoost Model', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Feature importance plot saved as 'feature_importance.png'")
    plt.close()
    
    # ========================================================================
    # SECTION 7: SHAP VALUES (Explainability)
    # ========================================================================
    print("\n" + "="*70)
    print("7. SHAP VALUES ANALYSIS (Model Explainability)")
    print("="*70)
    
    try:
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Risk Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        print("\n✅ SHAP summary plot saved as 'shap_summary_plot.png'")
        plt.close()
        
        # SHAP Bar Plot (Mean Absolute Impact)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=features, 
                         plot_type="bar", show=False)
        plt.title('SHAP Feature Importance - Mean Absolute Impact', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
        print("✅ SHAP bar plot saved as 'shap_bar_plot.png'")
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️  SHAP analysis warning: {e}")
    
    # ========================================================================
    # SECTION 8: DATASET STATISTICS
    # ========================================================================
    print("\n" + "="*70)
    print("8. DATASET STATISTICS")
    print("="*70)
    
    print(f"\nTotal Samples: {len(df)}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")
    print(f"\nClass Distribution:")
    print(f"  No Risk (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.2f}%)")
    print(f"  Risk (1):    {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.2f}%)")
    
    # Class distribution visualization
    plt.figure(figsize=(8, 6))
    class_counts = y.value_counts()
    plt.bar(['No Risk', 'Risk'], class_counts.values, color=['#2ecc71', '#e74c3c'])
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xlabel('Risk Category', fontsize=12)
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✅ Class distribution plot saved as 'class_distribution.png'")
    plt.close()
    
    # ========================================================================
    # SECTION 9: EXPORT METRICS TO CSV
    # ========================================================================
    print("\n" + "="*70)
    print("9. EXPORTING METRICS TO CSV")
    print("="*70)
    
    metrics_summary = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 
                   'Average Precision', 'CV Accuracy (Mean)', 'CV Accuracy (Std)'],
        'Value': [accuracy, precision, recall, f1, roc_auc, avg_precision, 
                  cv_scores.mean(), cv_scores.std()]
    }
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv('model_metrics_summary.csv', index=False)
    print("\n✅ Metrics summary exported to 'model_metrics_summary.csv'")
    
    # Export detailed results
    results_df = pd.DataFrame({
        'True_Label': y_test.values,
        'Predicted_Label': y_pred,
        'Risk_Probability': y_pred_proba
    })
    results_df.to_csv('detailed_predictions.csv', index=False)
    print("✅ Detailed predictions exported to 'detailed_predictions.csv'")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\n📊 Generated Files for Research Paper:")
    print("  1. confusion_matrix.png")
    print("  2. roc_curve.png")
    print("  3. precision_recall_curve.png")
    print("  4. cross_validation_scores.png")
    print("  5. feature_importance.png")
    print("  6. shap_summary_plot.png")
    print("  7. shap_bar_plot.png")
    print("  8. class_distribution.png")
    print("  9. model_metrics_summary.csv")
    print("  10. detailed_predictions.csv")
    print("\n✅ All metrics and visualizations ready for your research paper!\n")


if __name__ == "__main__":
    generate_comprehensive_metrics()
