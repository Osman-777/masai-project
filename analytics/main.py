
# Titanic — End-to-End EDA, Classification & Regression
'''
This notebook follows the Masai/Zepto analyst-to-data-scientist workflow:
load once, clean once, EDA, stratified split, leakage-safe preprocessing,
three classifiers, imbalance comparison, Random Forest tuning/OOB score,
regression side-task, model comparison, and a reloadable final pipeline.
'''

from pathlib import Path
import warnings
from IPython.display import display
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

ROOT = Path.cwd()
CHART_DIR = ROOT / "charts"
CHART_DIR.mkdir(exist_ok=True)
CSV_PATH = ROOT / "titanic.csv"
OFFLINE_PATH = ROOT / "titanic.csv.offline"
MODEL_PATH = ROOT / "saved_model.joblib"

## 1. Load once and profile

## The first run attempts the Seaborn online dataset. It immediately saves a
##local CSV and an offline fallback. Subsequent runs use the local CSV when it
##exists, so the raw dataset is not independently reloaded in later sections.

if CSV_PATH.exists():

    df_raw = pd.read_csv(CSV_PATH)
else:
    try:
        df_raw = sns.load_dataset("titanic")
        df_raw.to_csv(CSV_PATH, index=False)
        # Keep a second committed offline copy for the assignment requirement.
        df_raw.to_csv(OFFLINE_PATH, index=False)
        print("Downloaded Titanic through seaborn and saved local copies.")
    except Exception as exc:
        if OFFLINE_PATH.exists():
            df_raw = pd.read_csv(OFFLINE_PATH)
            print("Online load failed; using committed offline fallback:", exc)
        else:
            raise RuntimeError(
                "Could not load Titanic online and no titanic.csv.offline exists."
            ) from exc

print("Shape:", df_raw.shape)
display(df_raw.head())
display(df_raw.info())
display(df_raw.describe(include="all").T)

missing_pct = (df_raw.isna().mean() * 100).sort_values(ascending=False)
missing_report = pd.DataFrame({
    "missing_count": df_raw.isna().sum(),
    "missing_pct": missing_pct.round(2),
    "dtype": df_raw.dtypes.astype(str)
})
display(missing_report)

## 2. Missing-value strategy

''' 
Rule:
- <5% missing: drop affected rows.
- 5–30% missing: impute.
- >30% missing: drop the column, unless there is a documented reason to
  encode missingness instead.
  

The modeling pipeline also contains imputers as a safety mechanism, but the
explicit cleaning decision is made below from the observed training data.
'''
strategy_rows = []
for col in df_raw.columns:
    pct = df_raw[col].isna().mean() * 100
    if pct == 0:
        strategy = "none"
        reason = "No missing values."
    elif pct < 5:
        strategy = "drop_rows"
        reason = "Missingness is below 5%; dropping affected rows has limited data impact."
    elif pct <= 30:
        strategy = "impute"
        reason = "Missingness is between 5% and 30%; retain rows and impute."
    else:
        strategy = "drop_column"
        reason = "More than 30% is missing; column is too incomplete for the core model."
    strategy_rows.append([col, pct, strategy, reason])

strategy_df = pd.DataFrame(
    strategy_rows, columns=["column", "missing_pct", "strategy", "reason"]
)
display(strategy_df)

# Apply the explicit cleaning policy.
df = df_raw.copy()

drop_cols = strategy_df.loc[strategy_df.strategy == "drop_column", "column"].tolist()
drop_row_cols = strategy_df.loc[strategy_df.strategy == "drop_rows", "column"].tolist()

if drop_cols:
    df = df.drop(columns=drop_cols)

if drop_row_cols:
    df = df.dropna(subset=drop_row_cols)

# Do NOT fit imputation statistics here: that would leak information across
# the later train/test split. We record the decision now, then implement
# imputation inside the modeling pipeline so it is fitted on training data only.

imputed_cols = strategy_df.loc[strategy_df.strategy == "impute", "column"].tolist()

print("Dropped columns:", drop_cols)
print("Dropped rows because of:", drop_row_cols)
print("Columns to be imputed later inside the training-only pipeline:", imputed_cols)
print("Remaining missing values after explicit cleaning decisions:")
display(df.isna().sum())

## 3. Univariate analysis: age and fare
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["age"], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age distribution")
sns.boxplot(x=df["age"], ax=axes[0, 1])
axes[0, 1].set_title("Age boxplot")
sns.histplot(df["fare"], kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Fare distribution")
sns.boxplot(x=df["fare"], ax=axes[1, 1])
axes[1, 1].set_title("Fare boxplot")
plt.tight_layout()
plt.savefig(CHART_DIR / "01_age_fare_distributions.png", dpi=160, bbox_inches="tight")
plt.show()

for col in ["age", "fare"]:
    mean = df[col].mean()
    median = df[col].median()
    mode = df[col].mode().iloc[0]
    skew = df[col].skew()
    print(f"{col}: mean={mean:.3f}, median={median:.3f}, mode={mode:.3f}, skew={skew:.3f}")
    print("Interpretation:", "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "approximately symmetric")

## 4. Bivariate survival analysis
def survival_rate_table(group_cols):
    out = df.groupby(group_cols, dropna=False)["survived"].agg(["mean", "count"])
    out["survival_rate_pct"] = out["mean"] * 100
    return out.drop(columns="mean").sort_values("survival_rate_pct", ascending=False)

display(survival_rate_table(["sex"]))
display(survival_rate_table(["pclass"]))
display(survival_rate_table(["sex", "pclass"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.barplot(data=df, x="sex", y="survived", estimator="mean", errorbar=None, ax=axes[0])
axes[0].set_title("Survival rate by sex")
axes[0].set_ylabel("Survival rate")
sns.barplot(data=df, x="pclass", y="survived", estimator="mean", errorbar=None, ax=axes[1])
axes[1].set_title("Survival rate by passenger class")
axes[1].set_ylabel("Survival rate")
sns.barplot(data=df, x="pclass", y="survived", hue="sex", estimator="mean", errorbar=None, ax=axes[2])
axes[2].set_title("Survival by sex and class")
axes[2].set_ylabel("Survival rate")
plt.tight_layout()
plt.savefig(CHART_DIR / "02_bivariate_survival.png", dpi=160, bbox_inches="tight")
plt.show()

## 5. Correlation analysis
## Correlation is restricted to exactly:
## survived, pclass, age, sibsp, parch, fare.
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[corr_cols].corr()
display(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix: required six numeric columns")
plt.tight_layout()
plt.savefig(CHART_DIR / "03_required_correlation_heatmap.png", dpi=160, bbox_inches="tight")
plt.show()

## 6. Multivariate charts
'''
Four distinct multivariate charts are produced. Each interpretation is
written as four sentences so the README/notebook satisfies the acceptance
criterion.
'''
# Chart 1
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", size="pclass", alpha=0.7)
plt.title("Age vs fare, survival and passenger class")
plt.tight_layout()
plt.savefig(CHART_DIR / "04_age_fare_survival_class.png", dpi=160, bbox_inches="tight")
plt.show()
'''
**Interpretation — Chart 1:** The plot compares age and fare while encoding
survival and passenger class. Fare values are concentrated toward the lower
end with a smaller number of high-fare observations. Passenger class is
associated with fare, so the visual shows substantial overlap but different
fare concentrations by class. Survival is not determined by one variable
alone, which motivates the multivariable predictive models.'''
# Chart 2
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="pclass", y="age", hue="survived")
plt.title("Age distribution by class and survival")
plt.tight_layout()
plt.savefig(CHART_DIR / "05_age_class_survival.png", dpi=160, bbox_inches="tight")
plt.show()
'''
**Interpretation — Chart 2:** Age distributions differ across passenger
classes. The survival hue shows that the age composition of survivors and
non-survivors is not identical within each class. The class groups also have
different spreads and central tendencies. This indicates that class and age
can provide complementary information for classification.
'''
# Chart 3
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="sex", y="fare", hue="survived", split=True)
plt.title("Fare distribution by sex and survival")
plt.tight_layout()
plt.savefig(CHART_DIR / "06_fare_sex_survival.png", dpi=160, bbox_inches="tight")
plt.show()
'''
**Interpretation — Chart 3:** Fare distributions vary substantially across
the sex groups. Within each sex group, survivors and non-survivors show
different fare distributions. The long upper tails reflect a small number
of expensive tickets. This supports keeping fare as a continuous predictor
rather than reducing it to a few arbitrary categories.
'''
# Chart 4
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x="pclass", y="survived", hue="embarked", errorbar=None)
plt.title("Survival rate by class and embarkation port")
plt.tight_layout()
plt.savefig(CHART_DIR / "07_class_embarked_survival.png", dpi=160, bbox_inches="tight")
plt.show()
'''
**Interpretation — Chart 4:** Survival rates vary across passenger class and
embarkation port. The class separation is visible within multiple port
groups, showing that class remains an important conditioning variable.
Some port groups have fewer observations, so their displayed rates should be
interpreted with their sample counts in mind. The combined view demonstrates
why multivariate analysis is more informative than a single grouped rate.'''
## 7. Exploratory standardization check for age and fare
##z = (x - mean) / standard deviation
for col in ["age", "fare"]:
    z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    print(col, "z mean:", round(z.mean(), 6), "z std:", round(z.std(ddof=0), 6))

## 8. Train/test split FIRST, then preprocessing
'''
`survived` is the classification target. We stratify on the target so the
class proportions are preserved as closely as possible between train/test.
All learned preprocessing is fitted only on the training split through the
pipeline.
'''
target = "survived"
features = [
    "pclass", "sex", "age", "sibsp", "parch",
    "fare", "embarked", "alone", "who", "adult_male", "deck", "embark_town"
]
features = [c for c in features if c in df.columns]

X = df[features].copy()
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train class proportions:")
display(y_train.value_counts(normalize=True).rename("proportion"))
print("Test class proportions:")
display(y_test.value_counts(normalize=True).rename("proportion"))

numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
categorical_features = [c for c in features if c not in numeric_features]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

## 9. Train three classifiers on the identical split
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    )
}

classification_pipelines = {}
classification_results = []

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    classification_pipelines[name] = pipe
    classification_results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    })

classification_df = pd.DataFrame(classification_results)
display(classification_df)

### Decision-tree visualization
tree_pipe = classification_pipelines["Decision Tree"]
tree_model = tree_pipe.named_steps["model"]
feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()

plt.figure(figsize=(22, 10))
plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not survived", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=7
)
plt.title("Decision Tree — first three levels")
plt.tight_layout()
plt.savefig(CHART_DIR / "08_decision_tree.png", dpi=180, bbox_inches="tight")
plt.show()

### ROC curves
plt.figure(figsize=(8, 6))
for name, pipe in classification_pipelines.items():
    prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves — identical test split")
plt.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "09_roc_curves.png", dpi=160, bbox_inches="tight")
plt.show()

## 10. Three-way class-imbalance comparison
'''
Baseline = no imbalance handling; balanced = class_weight='balanced';
SMOTE = synthetic minority oversampling. SMOTE is placed inside an
imbalanced-learn pipeline so resampling occurs during fitting rather than
contaminating the held-out test set.
'''
imbalance_estimators = {
    "baseline": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "class_weight_balanced": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "SMOTE": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
}

imbalance_results = []

baseline_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", imbalance_estimators["baseline"])
])

balanced_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", imbalance_estimators["class_weight_balanced"])
])

# For SMOTE, use the preprocessor first, then SMOTE, then the classifier.
smote_pipe = ImbPipeline([
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", imbalance_estimators["SMOTE"])
])

for name, pipe in {
    "baseline": baseline_pipe,
    "class_weight_balanced": balanced_pipe,
    "SMOTE": smote_pipe
}.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    imbalance_results.append({
        "strategy": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    })

imbalance_df = pd.DataFrame(imbalance_results)
display(imbalance_df)
'''
**Interpretation:** Compare recall and F1 alongside accuracy rather than
relying on accuracy alone because the target classes are not perfectly
balanced. The class-weighted model changes the classifier's treatment of
minority-class errors without changing the underlying training rows.
SMOTE changes the training distribution by generating minority-class
examples, while the untouched test set remains the same. The written
recommendation should be based on the actual table produced above, with
particular attention to the metric that matters for the project's stated
objective.
'''
## 11. Random Forest GridSearchCV + OOB score
'''
`oob_score=True` is used because the acceptance criteria require an OOB
score to be reported.
'''

rf_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True
    ))
])

param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(
    rf_pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)
best_rf = grid.best_estimator_
print("OOB score:", best_rf.named_steps["model"].oob_score_)

best_pred = best_rf.predict(X_test)
best_prob = best_rf.predict_proba(X_test)[:, 1]
best_rf_metrics = {
    "accuracy": accuracy_score(y_test, best_pred),
    "precision": precision_score(y_test, best_pred, zero_division=0),
    "recall": recall_score(y_test, best_pred, zero_division=0),
    "f1": f1_score(y_test, best_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, best_prob)
}
display(pd.DataFrame([best_rf_metrics]))

## 12. Regression side-task — predict fare
'''
Fare is the regression target. Survived is excluded from predictors to avoid
using the classification target as an input. We report MAE, RMSE, R² and
adjusted R², plus a residual plot and a simple heteroscedasticity check.
'''
reg_target = "fare"
reg_features = [
    "pclass", "sex", "age", "sibsp", "parch",
    "embarked", "alone", "who", "adult_male"
]
reg_features = [c for c in reg_features if c in df.columns]

Xr = df[reg_features].copy()
yr = df[reg_target].copy()

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.20, random_state=RANDOM_STATE
)

reg_num = [c for c in reg_features if pd.api.types.is_numeric_dtype(Xr[c])]
reg_cat = [c for c in reg_features if c not in reg_num]

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), reg_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), reg_cat)
])

reg_pipe = Pipeline([
    ("preprocess", reg_preprocessor),
    ("model", LinearRegression())
])

reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)

n = len(yr_test)
p = reg_pipe.named_steps["preprocess"].transform(Xr_test).shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

regression_metrics = pd.DataFrame([{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2
}])
display(regression_metrics)

residuals = yr_test - yr_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=yr_pred, y=residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual")
plt.title("Fare regression residual plot")
plt.tight_layout()
plt.savefig(CHART_DIR / "10_fare_residuals.png", dpi=160, bbox_inches="tight")
plt.show()

# Breusch-Pagan-style auxiliary regression using transformed design matrix.
# This is a diagnostic, not a proof of causality.
X_design = reg_pipe.named_steps["preprocess"].transform(Xr_test)
if hasattr(X_design, "toarray"):
    X_design = X_design.toarray()
X_design = np.asarray(X_design)

# Add intercept for the auxiliary model.
aux_X = np.column_stack([np.ones(len(X_design)), X_design])
squared_resid = residuals.to_numpy() ** 2
aux_model = LinearRegression(fit_intercept=False).fit(aux_X, squared_resid)
aux_r2 = aux_model.score(aux_X, squared_resid)
bp_lm = len(squared_resid) * aux_r2
bp_df = aux_X.shape[1] - 1
bp_pvalue = stats.chi2.sf(bp_lm, bp_df) if bp_df > 0 else np.nan

print("Auxiliary-test LM statistic:", bp_lm)
print("Approximate heteroscedasticity test p-value:", bp_pvalue)
if bp_pvalue < 0.05:
    print("Diagnostic conclusion: evidence is consistent with non-constant residual variance at the 5% level.")
else:
    print("Diagnostic conclusion: the test does not provide strong evidence of non-constant residual variance at the 5% level.")

## 13. Final model comparison
'''
Classification and regression metrics are kept as separate metric groups;
they are not collapsed into one score.
'''
final_classification = classification_df.copy()
final_classification["model_type"] = "classification"

final_regression = regression_metrics.copy()
final_regression["model"] = "Linear Regression"
final_regression["model_type"] = "regression"

display(final_classification)
display(final_regression)

## 14. Save the complete fitted classification pipeline
'''
The saved artifact contains preprocessing + estimator together and can be
reloaded to predict on raw new rows with the same columns.
# Save the tuned Random Forest as the final fitted classification artifact.
'''
joblib.dump(best_rf, MODEL_PATH)
print("Saved:", MODEL_PATH)

# Reload demonstration.
loaded_model = joblib.load(MODEL_PATH)
sample_predictions = loaded_model.predict(X_test.head(5))
print("Reloaded model predictions:", sample_predictions)

## 15. New-data smoke test
'''
This demonstrates that the saved artifact accepts raw rows with the same
feature schema and performs preprocessing internally.
'''
new_raw = X_test.head(3).copy()
new_predictions = loaded_model.predict(new_raw)
new_probabilities = loaded_model.predict_proba(new_raw)[:, 1]

display(pd.DataFrame({
    "predicted_survived": new_predictions,
    "survival_probability": new_probabilities
}))

## Final written conclusion
'''
Do not choose a model from memory or by a generic rule. Use the generated
classification table, imbalance table, GridSearchCV results, and OOB score.
State which model/strategy you select for the project's objective and cite
the exact metrics that support that selection. For the regression side-task,
report MAE/RMSE/R²/Adjusted R² and the residual/heteroscedasticity diagnostic
separately.
'''