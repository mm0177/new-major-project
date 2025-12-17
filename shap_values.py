import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Load the model and fused data
model = joblib.load('risk_engine.pkl')
df_final = pd.read_csv('final_fused_dataset.csv')
features = ['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']
X = df_final[features]

# Generate and Save SHAP Plot
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# This creates a Summary Plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig('shap_summary_report.png', bbox_inches='tight')
print("✅ SHAP Summary Plot saved as 'shap_summary_report.png'")