Employee Attrition Prediction

Predicting which employees are likely to leave the company, using a logistic regression that is calibrated, interpretable and built with a fairness check in mind. The goal is not just a good AUC, but a model an HR team could actually act on: probabilities that mean what they say, a decision threshold chosen from business costs, and no reliance on sensitive personal attributes.

Academic project for the MSc in Big Data, Business Analytics & AI (ESIC). Code comments and the notebook are written in Spanish.

Results

Held-out test set: 441 employees, 71 of whom left the company.

Metric	Value	In plain terms
ROC AUC	0.824	Ranks a leaver above a stayer 82% of the time
Recall	70.4%	50 of the 71 leavers are flagged
Precision	49.0%	About 1 in 2 alarms is a real leaver
Specificity	86.0%	86% of stayers are correctly left alone
Alarms raised	102	Out of 441 employees (23%)
Brier score	0.098	vs. 0.135 for always predicting the base rate

Cross-validation on the training set (5-fold stratified, repeated 10 times): AUC 0.865, Brier 0.084.

Accuracy is deliberately not reported. Only 16% of employees leave, so a model that always predicts "stays" would already reach 84%.

Key decisions
No class reweighting. class_weight="balanced" gave the same AUC but inflated the predicted probabilities (mean 0.34 vs. a real attrition rate of 0.15). The class imbalance is handled by lowering the decision threshold instead, so the probabilities stay calibrated.
Cost-based threshold (0.23). A missed leaver costs roughly 20% of an annual salary in replacement costs, while a false alarm costs one conversation with a manager. The threshold targets ~70% recall while keeping the number of alarms manageable.
Calibration checked, not assumed: calibration curve by quantile bins plus Brier score.
Sensitive attributes excluded. Gender, Age and MaritalStatus were analysed in the EDA, and the model was compared with and without them. The final model leaves them out: AI used for employee management is classified as high-risk under the EU AI Act.
Interpretability. Odds ratios for the coefficients, plus permutation importance (30+ repeats) computed on the original variables so that one-hot dummies are grouped back together.
Feature engineering with business meaning, implemented as custom scikit-learn transformers:
SalarioRelativo: salary relative to the median of the same job level and role, computed on the training data only. It replaces MonthlyIncome and removes its collinearity with JobLevel (VIF > 10).
NumInsatisfaccion: number of satisfaction scales at their lowest level. It replaces the four separate satisfaction scales.
TieneStockOptions: binary flag, since the effect of StockOptionLevel is not linear.
Log transform of the skewed tenure variables.
No data leakage. All preprocessing lives inside a single scikit-learn Pipeline, fitted within each cross-validation fold.
Monte Carlo simulation of the expected number of leavers per department, built on the calibrated probabilities.
Project structure
├── config.yaml              # Paths, seed, columns to drop, MLflow settings
├── requirements.txt         # Pinned direct dependencies
├── data/raw/                # Original dataset (never modified)
├── notebooks/
│   └── analisis.ipynb       # EDA, modelling, evaluation and conclusions
├── src/
│   ├── data.py              # Config and data loading with validation checks
│   ├── features.py          # Custom sklearn transformers (feature engineering)
│   ├── pipeline.py          # Preprocessing + model in a single Pipeline
│   └── evaluate.py          # Repeated stratified CV and MLflow tracking
├── models/
│   ├── modelo_final.joblib  # Trained pipeline
│   └── metadatos.json       # Chosen variant, threshold and test metrics
└── reports/figures/         # All figures used in the report
How to reproduce
bash
git clone https://github.com/Yeray-Martinez-Acha/employee-attrition-prediction.git
cd employee-attrition-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

Open notebooks/analisis.ipynb and run all cells. The random seed is fixed in config.yaml, so the results are reproducible.

Experiments are logged to a local SQLite database (mlflow.db). To browse them, run this from the project root:

bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
Limitations
The dataset is fictional, created by IBM data scientists, and it is unknown when each variable was recorded.
Prediction is not causation. The model flags risk; it does not explain why a person leaves.
It does not distinguish between healthy and harmful turnover (PerformanceRating only takes the values 3 and 4).
An alarm should trigger support, not penalties. Otherwise the model risks becoming a self-fulfilling prophecy.
Data

IBM HR Analytics Employee Attrition & Performance: 1,470 employees, 35 variables.

Tech stack

Python · pandas · scikit-learn · statsmodels · matplotlib · seaborn · MLflow