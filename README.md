
Titanic Survival Prediction Project
-----------------------------------

Author: Adonay Eshetayehu Tefera

Project Overview:
- Predict whether a passenger survived the Titanic disaster using multiple machine learning models.
- Models included: Random Forest, Logistic Regression, SVC, XGBoost, LightGBM.
- Preprocessing includes handling missing values, scaling numerical features, and one-hot encoding categorical features.

Workflow:
1. Load data from seaborn's Titanic dataset.
2. Preprocess features using ColumnTransformer and Pipelines.
3. Train multiple models using GridSearchCV for hyperparameter tuning.
4. Evaluate models:
   - Metrics: Accuracy, F1-score, Precision, Recall
   - Confusion matrices saved as PNG
   - Classification reports saved as TXT
   - Accuracy comparison bar chart
   - Model comparison table saved as CSV
5. All outputs are saved in the `outputs/` directory for reproducibility.

Usage:
- Ensure all dependencies are installed (see requirements.txt).
- Run the main workflow:
    python run.py
- Check `outputs/` folder for all evaluation results and plots.

Dependencies:
- numpy, pandas, scikit-learn, matplotlib, seaborn, xgboost, lightgbm

### Notes
- This project implements a **full machine learning pipeline**:
  - Data loading and splitting
  - Preprocessing (imputation, scaling, one-hot encoding)
  - Model training and hyperparameter tuning
  - Evaluation and comparison of multiple models
- Includes 5 classifiers: Random Forest, Logistic Regression, SVC, XGBoost, LightGBM
- Provides visualizations: confusion matrices, feature importances, and coefficient magnitudes
- Modular structure for clarity and reusability
