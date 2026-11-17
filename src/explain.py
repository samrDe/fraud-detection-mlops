"""Generate global SHAP feature-importance plot for the trained tree model."""
from pathlib import Path
import joblib
import pandas as pd
import shap
from src.data import FEATURES

ROOT = Path(__file__).resolve().parents[1]
bundle = joblib.load(ROOT / "artifacts/model.joblib")
test = pd.read_csv(ROOT / "data/processed/test.csv")
X = test[FEATURES].sample(n=min(500, len(test)), random_state=42)
explainer = shap.TreeExplainer(bundle["model"])
values = explainer.shap_values(X)
# Binary tree classifiers may return a list (one array per class).
positive_class_values = values[1] if isinstance(values, list) else values
ROOT.joinpath("reports").mkdir(exist_ok=True)
shap.summary_plot(positive_class_values, X, show=False, plot_type="bar")
shap.plt.savefig(ROOT / "reports/shap_summary.png", bbox_inches="tight", dpi=160)
shap.plt.close()
print("Saved reports/shap_summary.png")
