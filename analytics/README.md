# Titanic Analytics Submission

## Files
- `main.py` — complete end-to-end notebook.
- `titanic_end_to_end.py` — same workflow as a Python script.
- `charts/` — generated chart destination.
- `titanic.csv` — created on first successful online Seaborn load.
- `titanic.csv.offline` — created alongside the CSV as the offline fallback.
- `saved_model.joblib` — created after fitting the tuned Random Forest.

## Run
1. Open the notebook in Jupyter/VS Code/Google Colab.
2. Install dependencies if necessary:
   `pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib scipy`
3. Run cells from top to bottom.
4. The first run attempts `sns.load_dataset("titanic")`, then saves local copies.
5. Later runs use the local `titanic.csv` rather than independently reloading the raw dataset.
6. The final `saved_model.joblib` contains preprocessing plus the fitted estimator and is demonstrated on raw test rows.

## Important
The notebook intentionally does not hard-code model metrics. Run it to generate the actual metrics from the fixed random seed and test split. The final written recommendation should use those generated values.
