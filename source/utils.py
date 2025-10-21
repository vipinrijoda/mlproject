import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from source.exception import CustomException
from source.logger import logging

def save_object(file_path, obj):
    """
    Save a Python object to a file using dill
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids=None):
    """
    Evaluate multiple machine learning models with hyperparameter tuning
    """
    try:
        report = {}

        for model_name, model in models.items():
            # Check if hyperparameter tuning is requested for this model
            if param_grids and model_name in param_grids:
                logging.info(f"Performing hyperparameter tuning for {model_name}")
                
                grid_search = RandomizedSearchCV(
                    model, 
                    param_grids[model_name], 
                    n_iter=10,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=42
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get the best model (already fitted during grid search)
                best_model = grid_search.best_estimator_
                
            else:
                # Use default model without tuning
                best_model = model
                best_model.fit(X_train, y_train)

            # Make predictions with best model
            y_test_pred = best_model.predict(X_test)

            # Calculate R2 score
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a file
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)