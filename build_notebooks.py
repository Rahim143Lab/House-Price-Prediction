"""
Generates the three project notebooks using nbformat, then executes them
with nbclient so that saved outputs (tables, plots) are real, not fabricated.

Run with: python build_notebooks.py
"""
import nbformat as nbf
from nbclient import NotebookClient

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)


# ============================================================
# 01_data_exploration.ipynb
# ============================================================
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md("# 01 · Data Exploration\n\n"
       "Exploratory data analysis of the Ames Housing dataset used by this "
       "project. This notebook mirrors the logic in `src/data_loader.py` "
       "and `src/eda.py` — no duplicated preprocessing logic is written here."),
    code("import sys\nsys.path.insert(0, '..')\n\n"
         "import pandas as pd\n"
         "from src import config, data_loader, eda\n\n"
         "pd.set_option('display.max_columns', 30)"),
    md("## Load and clean the raw dataset"),
    code("df = data_loader.load_raw_data()\n"
         "df = data_loader.basic_clean(df)\n"
         "df.shape"),
    code("df.head()"),
    md("## Dataset summary\n\nShape, dtypes, missing values, duplicates, skewness."),
    code("summary = eda.summarize_dataset(df)\nsummary"),
    code("df.dtypes.value_counts()"),
    code("missing = df.isna().sum().sort_values(ascending=False)\n"
         "missing[missing > 0].head(15)"),
    md("## Target variable: SalePrice\n\n"
       "The target is right-skewed, as is typical for house prices — most "
       "homes cluster in a moderate price range with a long tail of "
       "expensive properties."),
    code("df[config.TARGET_COLUMN].describe()"),
    code("eda.plot_target_distribution(df, config.FIGURES_DIR)\n"
         "print('Saved to', config.FIGURES_DIR / 'target_distribution.png')"),
    md("![target distribution](../reports/figures/target_distribution.png)"),
    md("## Correlations with key numeric features"),
    code("eda.plot_correlation_heatmap(df, config.FIGURES_DIR)\n"
         "print('Saved to', config.FIGURES_DIR / 'correlation_heatmap.png')"),
    md("![correlation heatmap](../reports/figures/correlation_heatmap.png)"),
    md("## Price vs. living area, bedrooms, bathrooms, age, and location"),
    code("eda.plot_price_vs_living_area(df, config.FIGURES_DIR)\n"
         "eda.plot_price_vs_bedrooms(df, config.FIGURES_DIR)\n"
         "eda.plot_price_vs_bathrooms(df, config.FIGURES_DIR)\n"
         "eda.plot_price_vs_property_age(df, config.FIGURES_DIR)\n"
         "eda.plot_price_by_location(df, config.FIGURES_DIR)\n"
         "eda.plot_boxplots_numeric(df, config.FIGURES_DIR)\n"
         "print('Saved all remaining EDA figures to', config.FIGURES_DIR)"),
    md("![price vs living area](../reports/figures/price_vs_living_area.png)\n"
       "![price vs bedrooms](../reports/figures/price_vs_bedrooms.png)\n"
       "![price vs bathrooms](../reports/figures/price_vs_bathrooms.png)\n"
       "![price vs property age](../reports/figures/price_vs_property_age.png)\n"
       "![price by location](../reports/figures/price_by_location.png)\n"
       "![boxplots](../reports/figures/boxplots_numeric_features.png)"),
    md("## Takeaways\n\n"
       "- `Overall Qual`, `Gr Liv Area`, `Garage Cars`, and total bathrooms show "
       "the strongest visual relationship with `SalePrice`.\n"
       "- `SalePrice` is right-skewed; tree-based models handle this natively, "
       "while linear models could benefit from a log-transformed target "
       "(left as a future improvement).\n"
       "- Several columns (`Pool QC`, `Misc Feature`, `Alley`, `Fence`) are "
       "mostly missing because the feature is genuinely absent for most "
       "homes (e.g. no pool) — these columns are intentionally excluded from "
       "the curated feature set used for modeling."),
]

# ============================================================
# 02_model_training.ipynb
# ============================================================
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md("# 02 · Model Training\n\n"
       "Trains and compares multiple regression models. This notebook calls "
       "the exact same functions used by `python -m src.train`, so results "
       "here match the production training script."),
    code("import sys\nsys.path.insert(0, '..')\n\n"
         "import pandas as pd\n"
         "from src import config, data_loader, feature_engineering, train as train_module"),
    md("## Load data and split (train / validation / test)"),
    code("train_df, val_df, test_df = data_loader.load_and_split()\n"
         "len(train_df), len(val_df), len(test_df)"),
    md("## Feature engineering"),
    code("X_train, y_train = train_module._prepare_xy(train_df)\n"
         "X_val, y_val = train_module._prepare_xy(val_df)\n"
         "X_train.head()"),
    md("## Train and compare baseline models\n\n"
       "Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, "
       "and XGBoost, each wrapped in the same preprocessing pipeline "
       "(`src/preprocessing.py`) to avoid preprocessing leakage."),
    code("comparison_df, fitted_pipelines = train_module.train_and_compare(\n"
         "    X_train, y_train, X_val, y_val\n"
         ")\n"
         "comparison_df"),
    md("## Hyperparameter tuning\n\n"
       "The strongest tree-based model from the comparison above is tuned "
       "with `RandomizedSearchCV` (5-fold cross-validation on the training "
       "set only)."),
    code("tree_models = comparison_df[comparison_df['Model'].isin(\n"
         "    ['Random Forest', 'Gradient Boosting', 'XGBoost']\n"
         ")]\n"
         "best_tree_model_name = tree_models.iloc[0]['Model']\n"
         "best_tree_model_name"),
    code("best_estimator, best_params, best_cv_score = train_module.tune_best_tree_model(\n"
         "    best_tree_model_name, X_train, y_train\n"
         ")\n"
         "best_params"),
    code("tuned_val_pred = best_estimator.predict(X_val)\n"
         "from src import utils\n"
         "utils.compute_regression_metrics(y_val, tuned_val_pred)"),
    md("## Note\n\n"
       "The full pipeline — including final model selection, retraining on "
       "train+validation, one-time test-set evaluation, and saving the model "
       "artifact — is orchestrated by `src/train.py`. Run it directly with:\n\n"
       "```bash\n"
       "python -m src.train\n"
       "```"),
]

# ============================================================
# 03_model_evaluation.ipynb
# ============================================================
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    md("# 03 · Model Evaluation\n\n"
       "Loads the final trained model saved by `python -m src.train` and "
       "evaluates it on the untouched test set. Mirrors `src/evaluate.py`."),
    code("import sys\nsys.path.insert(0, '..')\n\n"
         "from src import config, data_loader, feature_engineering, evaluate, utils"),
    md("## Load the final trained pipeline"),
    code("pipeline = evaluate.load_model()\npipeline"),
    md("## Predict on the held-out test set"),
    code("_, _, test_df = data_loader.load_and_split(save=False)\n"
         "X_test = feature_engineering.build_features(test_df)\n"
         "y_test = test_df[config.TARGET_COLUMN].values\n"
         "y_pred = pipeline.predict(X_test)"),
    md("## Test set metrics"),
    code("metrics = utils.compute_regression_metrics(y_test, y_pred)\nmetrics"),
    md("## Actual vs. predicted, and residual distribution"),
    code("evaluate.plot_actual_vs_predicted(y_test, y_pred, config.FIGURES_DIR)\n"
         "evaluate.plot_residual_distribution(y_test, y_pred, config.FIGURES_DIR)\n"
         "print('Saved figures to', config.FIGURES_DIR)"),
    md("![actual vs predicted](../reports/figures/actual_vs_predicted.png)\n"
       "![residual distribution](../reports/figures/residual_distribution.png)"),
    md("## Feature importance"),
    code("importance_df = evaluate.get_feature_importance(pipeline)\n"
         "importance_df.head(15)"),
    code("evaluate.plot_feature_importance(importance_df, config.FIGURES_DIR)\n"
         "print('Saved to', config.FIGURES_DIR / 'feature_importance.png')"),
    md("![feature importance](../reports/figures/feature_importance.png)"),
    md("## Summary\n\n"
       "The metrics, plots, and feature importance above are generated live "
       "from the currently saved model artifact — nothing here is "
       "hard-coded. If you retrain the model, re-running this notebook will "
       "reflect the new results."),
]

notebooks = {
    "notebooks/01_data_exploration.ipynb": nb1,
    "notebooks/02_model_training.ipynb": nb2,
    "notebooks/03_model_evaluation.ipynb": nb3,
}

for path, nb in notebooks.items():
    nbf.write(nb, path)
    print("Wrote", path)

for path, nb in notebooks.items():
    print("Executing", path, "...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nbf.write(nb, path)
    print("Executed and saved", path)
