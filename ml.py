from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import os
import pandas as pd
import joblib
from tda_features import get_tda_features_and_labels
import numpy as np 

def train_and_tune_model(model, X_train, y_train, X_val, y_val, param_grid, model_name, scoring='f1_weighted', threshold=0.3):
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1, return_train_score=True, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    model_path = f"models/{model_name}_best_model.pkl"
    joblib.dump(best_model, model_path)

    y_val_pred = best_model.predict(X_val)



    metrics = {
        "model": model_name,
        "best_params": grid_search.best_params_, 
        "f1_score": f1_score(y_val, y_val_pred, average='weighted'),
        "accuracy" : accuracy_score(y_val, y_val_pred),
        "precision" : precision_score(y_val, y_val_pred, average='weighted'),
        "recall" : recall_score(y_val, y_val_pred, average='weighted'),
    }

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(f"metrics/{model_name}_grid_search_results.csv", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_summary_path = f"metrics/{model_name}_metrics_summary.csv"
    metrics_df.to_csv(metrics_summary_path, index=False)

    return best_model, metrics

def run_catboost_classifier(data, model_name=None, threshold=0.3):
    param_grid = {
        "iterations": [100, 200],
        "depth": [3, 6],
        "learning_rate": [0.01, 0.1]
    }

    X_train, y_train, X_val, y_val,  X_test, y_test, = get_tda_features_and_labels(data)

    best_model, metrics = train_and_tune_model(
        CatBoostClassifier(verbose=0),
        X_train, y_train, X_val, y_val,
        param_grid,
        model_name
    )

    y_test_pred = best_model.predict(X_test)
    classes = [str(c) for c in np.unique(y_test)]
    print("\nTest Results:")
    print(generate_classification_report(y_test, y_test_pred, classes))


    return best_model, metrics


def generate_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, zero_division=1, target_names=classes)
    return report