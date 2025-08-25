https://github.com/rokmark0/auto-feature-engineering/releases

# 🚀 Auto Feature Engineering — Generate Robust ML Features Fast
Automated feature engineering toolkit for tabular and time-series tasks. Use multiple strategies: statistical transforms, polynomial interactions, genetic programming, and time-series aggregations. Works with pandas, scikit-learn, and common ML pipelines.

[![Releases](https://img.shields.io/badge/Releases-View-blue)](https://github.com/rokmark0/auto-feature-engineering/releases)

![Data pipeline illustration](https://images.unsplash.com/photo-1555949963-aa79dcee981d?auto=format&fit=crop&w=1500&q=80)

About
- Purpose: Reduce manual feature design work. Generate candidate features, score them, and return a compact set ready for modeling.
- Scope: Tabular data, time-series, classification, regression.
- Key methods: autofeat-style linear transforms, gplearn genetic programs, tsfresh-like time-series features, featuretools-like deep feature synthesis.

Get the release asset
- This repository provides release builds at the link above.
- Download the release file and execute it to install or run the packaged tool.
- Example: download the asset named auto-feature-engineering-v1.2.0.tar.gz and extract or run the provided installer script.

Why this tool
- Save time designing features.
- Use multiple algorithms and cross-validate candidate features.
- Keep a small, useful set that integrates with scikit-learn.
- Track feature provenance for reproducibility.

Topics and techniques
- autofeat, autofeatures, feature-engineering, feature-extraction
- featuretools (deep feature synthesis)
- genetic-algorithm, genetic-algorithms, gplearn
- time-series, tsfresh

Core capabilities
- Column transforms: log, sqrt, rank, scaling.
- Interaction generation: pairwise product, ratio, difference.
- Polynomial expansion up to degree 3 with automatic selection.
- Genetic programming: create interpretable symbolic features (gplearn).
- Time-series aggregations: sliding window stats, FFT, seasonal indicators.
- Feature selection: L1, tree-based importance, mutual information, greedy selection.
- Pipeline export: scikit-learn Pipeline, ONNX export support.
- Provenance: maintain recipe for each feature, including seed transforms and algorithm parameters.

Quick start — install from release
1. Visit the Releases page and download the release file:
   - https://github.com/rokmark0/auto-feature-engineering/releases
2. Extract and run the installer or run the executable asset.
   - Example commands:
```
# Replace with actual release file name after download
tar -xzf auto-feature-engineering-v1.2.0.tar.gz
cd auto-feature-engineering
./install.sh
```
3. After install you can run the CLI or import the Python package.

Quick start — pip-like install from release archive
```
# If the release provides a wheel or tarball
pip install auto_feature_engineering-1.2.0-py3-none-any.whl
```

Basic Python usage
```
from afe import AutoFeatureEngineering

afe = AutoFeatureEngineering(
    target_col='price',
    task='regression',
    time_col=None,
    max_features=100,
    methods=['stat', 'poly', 'genetic']
)

# X is a pandas DataFrame, y is a Series
afe.fit(X, y)
X_new = afe.transform(X)
# X_new contains engineered features plus provenance metadata
```

CLI
- Run feature generation without writing code.
```
afe-cli generate \
  --input data/train.csv \
  --target price \
  --task regression \
  --output data/train_features.csv \
  --methods stat,poly,genetic \
  --max-features 80
```

Time-series example
```
afe = AutoFeatureEngineering(
    target_col='demand',
    task='forecast',
    time_col='timestamp',
    freq='H',
    methods=['ts_rolling','fft','seasonal']
)
afe.fit(df)
df_feat = afe.transform(df)
```

Genetic programming (gplearn) notes
- The tool uses a compact GP search to propose symbolic features.
- You can restrict operators to keep expressions interpretable:
```
gparams = {
  'population_size': 300,
  'generations': 10,
  'function_set': ['add','sub','mul','div']
}
afe.set_gplearn_params(gparams)
```

Feature selection and evaluation
- The tool ranks candidates with cross-validation.
- Selectors available: Lasso, RandomForest feature importance, mutual information, greedy forward selection.
- Each selected feature keeps a score and a provenance string.
- Export selection report as CSV or JSON.

Integration with ML pipelines
- The transform returns a DataFrame.
- Use with scikit-learn Pipeline:
```
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([
  ('afe', afe),
  ('clf', RandomForestRegressor(n_estimators=100))
])
pipe.fit(X_train, y_train)
```

Export and reproducibility
- Save the feature recipe:
```
afe.save_recipe('recipe.json')
```
- Load and reuse:
```
afe = AutoFeatureEngineering.load_recipe('recipe.json')
X_new = afe.transform(X)
```
- Export model input pipeline to ONNX:
```
afe.to_onnx('pipeline.onnx', sample_input=X.head(1))
```

Configuration options
- max_features: max number of final features.
- methods: list of methods to run (stat, poly, cross, genetic, ts_rolling, fft).
- time_col: column name for timestamps for time-series features.
- scoring: metric for selection (rmse, mae, f1, roc_auc).
- feature_filter: automatic filter to remove low-variance or constant features.

Best practices
- Run with a small max_features first to check results.
- Use cross-validation to avoid overfitting on generated features.
- Inspect provenance and test new features on holdout sets.
- For time-series, respect temporal folds for cross-validation.

Examples and recipes
- Retail demand forecasting: use ts_rolling and seasonal methods, include lag and rolling stats.
- Fraud detection: use gplearn to create non-linear ratios and interactions.
- Pricing model: combine polynomial features with L1 selection to keep interpretability.

Benchmarks
- In tests on public datasets, the tool improved baseline models for structured tasks by generating complementary features.
- Use the selection report to compare custom preprocessing vs generated features.

Visualization
- The tool includes a small dashboard to inspect:
  - Feature importance histogram.
  - Score trace for feature generation runs.
  - Feature expression viewer for GP outputs.
- Run the dashboard:
```
afe-ui serve --port 8080 --recipe recipe.json
```

Extending the tool
- Add new transform modules in afe/transforms.
- Add custom feature scorers by implementing the scorer API.
- Plug new selection strategies via the selection registry.

Contributing
- Fork the repository, add tests for new features, and open a pull request.
- Follow the existing code style and add docs for new modules.
- Run the unit tests:
```
pytest tests/
```

Release artifacts and automation
- The Releases page contains binaries, installers, and source archives.
- The release asset may include:
  - Wheels for different Python versions.
  - A portable binary for Linux/Mac/Windows.
  - A recipe JSON example.
- Download the asset and execute install steps described in the archive.

Security and sandbox
- The tool runs transforms on local data.
- Review generated expressions from GP before use in production.
- Use reproducible seeds for GP runs to help auditing.

License
- MIT license. Check LICENSE file for details.

References and related projects
- autofeat — automatic linear feature generation.
- featuretools — deep feature synthesis.
- gplearn — genetic programming for symbolic regression.
- tsfresh — extraction of time-series characteristics.

Contact and support
- Open issues on GitHub for bug reports or feature requests.
- Use Discussions for usage questions and recipe sharing.
- For binary downloads and releases, use the Releases page:
  - https://github.com/rokmark0/auto-feature-engineering/releases

Badges
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/rokmark0/auto-feature-engineering/ci.yml?branch=main)](https://github.com/rokmark0/auto-feature-engineering/actions)

---

Screenshots
- Feature list view:
  ![Feature list](https://raw.githubusercontent.com/rokmark0/auto-feature-engineering/main/docs/images/features.png)
- GP expression viewer:
  ![GP expressions](https://raw.githubusercontent.com/rokmark0/auto-feature-engineering/main/docs/images/gp_view.png)