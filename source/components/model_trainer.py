import os
import sys
from dataclasses import dataclass

# Machine Learning models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom modules
from source.exception import CustomException
from source.logger import logging
from source.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_hyperparameter_grids(self):
        """
        Define hyperparameter grids for each model
        """
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            "Decision Tree": {
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 6],
                'criterion': ['squared_error', 'friedman_mse']
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "Linear Regression": {
                'fit_intercept': [True, False],
                'copy_X': [True, False]
            },
            "K-Neighbors Regressor": {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            "XGBRegressor": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            "CatBoost Regressor": {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5, 7]
            },
            "AdaBoost Regressor": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                'loss': ['linear', 'square', 'exponential']
            }
        }
        return param_grids

    def initiate_model_trainer(self, train_array, test_array): 
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(random_state=42),
                "CatBoost Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42)
            }

            # Get hyperparameter grids for tuning
            param_grids = self.get_hyperparameter_grids()

            # Evaluate models with hyperparameter tuning
            model_report = evaluate_models(
                X_train=x_train, 
                y_train=y_train, 
                X_test=x_test, 
                y_test=y_test, 
                models=models,
                param_grids=param_grids
            )
            
            # Find best model based on test score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Get the best model instance and fit it on the full training data
            best_model = models[best_model_name]
            
            # FIX: Fit the best model on training data before making predictions
            best_model.fit(x_train, y_train)

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            
            logging.info(f"Best found model: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)