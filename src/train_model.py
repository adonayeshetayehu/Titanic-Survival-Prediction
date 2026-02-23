# Example inside train_model.py
def train_model(preprocessor, X_train, y_train, model_name="RandomForest"):
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    if model_name == "RandomForest":
        classifier = RandomForestClassifier(random_state=42)
        param_grid = {
            'classifier__n_estimators':[50,100,200],
            'classifier__max_depth':[None,10,20],
            'classifier__min_samples_split':[2,5,10]
        }
    elif model_name == "LogisticRegression":
        classifier = LogisticRegression(random_state=42)
        param_grid = {
            'classifier__solver':['liblinear', 'saga'],
            'classifier__penalty':['l1','l2', 'elasticnet'],
            'classifier__class_weight':[None,'balanced']
        }
    elif model_name == "SVC":
        classifier = SVC(random_state=42)
        param_grid = {
            'classifier__C':[0.1,1,10, 50],
            'classifier__kernel':['linear','rbf', 'poly'],
            'classifier__gamma':['scale','auto'],
        }
    elif model_name == "XGBoost":
        classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_grid = {
            'classifier__n_estimators':[50,100,200],
            'classifier__max_depth':[3,5,7],
            'classifier__learning_rate':[0.01,0.05],
            'classifier__subsample':[0.6,0.8],
            'classifier__colsample_bytree':[0.6,0.8]
        }
    elif model_name == "LightGBM":
        classifier = LGBMClassifier(random_state=42)
        param_grid = {
            'classifier__n_estimators':[50,100,200],
            'classifier__max_depth':[3,5,7],
            'classifier__learning_rate':[0.01,0.05,0.1],
            'classifier__num_leaves':[15,31,63],
            'classifier__subsample':[0.6,0.8],
            'classifier__colsample_bytree':[0.6,0.8]
        }
    else:
        raise ValueError(f"Unknown model_name {model_name}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    
    model = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=0)
    model.fit(X_train, y_train)
    return model