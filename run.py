# run.py
from src.data import load_data
from src.preprocess import build_preprocessor
from src.train_model import train_model  # single flexible function for all models
from src.evaluate import evaluate_models

# 1. Load data
X_train, X_test, y_train, y_test = load_data()

# 2. Build preprocessor
preprocessor, numerical_features, categorical_features = build_preprocessor(X_train)

# 3. Define models to train
model_names = ["RandomForest", "LogisticRegression", "SVC", "XGBoost", "LightGBM"]
models = {}

# 4. Train each model
for name in model_names:
    print(f"\nTraining {name}...")
    model = train_model(preprocessor, X_train, y_train, model_name=name)
    models[name] = model

# 5. Evaluate all models and save outputs
results_df = evaluate_models(models, X_test, y_test)

# 6. Print comparison table
print("\n=== Model Comparison ===")
print(results_df)