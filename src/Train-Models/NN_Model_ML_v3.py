import sqlite3
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
import xgboost as xgb
import optuna
import shap
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load Data
def load_data(db_path, table_name):
    """
    Load data from a SQLite database.

    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to query.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        con = sqlite3.connect(db_path)
        query = f'SELECT * FROM "{table_name}"'
        data = pd.read_sql_query(query, con, index_col="index")
        con.close()
        print(f"Data loaded successfully from table '{table_name}'.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Feature Selection based on Importance
def select_important_features(data, importance_list, top_n=20):
    """
    Select top-N important features from the data based on the importance list.

    Args:
        data (pd.DataFrame): The dataset.
        importance_list (list of tuples): List containing tuples of (feature_name, importance_score).
        top_n (int): Number of top features to select.

    Returns:
        pd.DataFrame: DataFrame containing only the top-N important features.
    """
    # Sort the importance list by score in descending order
    sorted_importance = sorted(importance_list, key=lambda x: x[1], reverse=True)
    top_features = [feature for feature, score in sorted_importance[:top_n]]
    print(f"Selected top-{top_n} important features: {top_features}")
    
    # Handle interaction features (features with spaces)
    interaction_features = []
    for feature in top_features.copy():
        if ' ' in feature:
            parts = feature.split(' ')
            if len(parts) == 2:
                new_feature_name = f"{parts[0]}_{parts[1]}"
                # Check if both parts exist in the data
                if parts[0] in data.columns and parts[1] in data.columns:
                    data[new_feature_name] = data[parts[0]] * data[parts[1]]
                    interaction_features.append(new_feature_name)
                    print(f"Created interaction feature: {new_feature_name}")
                    # Replace the original feature with the new interaction feature
                    top_features[top_features.index(feature)] = new_feature_name
                else:
                    print(f"Cannot create interaction feature '{new_feature_name}' as one of the components is missing.")
                    # Remove the feature if components are missing
                    top_features.remove(feature)
    
    # Replace spaces with underscores in feature names
    top_features = [f.replace(' ', '_') for f in top_features]
    
    # Ensure all selected features exist in the data
    missing_features = [feat for feat in top_features if feat not in data.columns]
    if missing_features:
        print(f"Warning: The following selected features are not in the dataset and will be ignored: {missing_features}")
        # Remove missing features from the list
        top_features = [feat for feat in top_features if feat in data.columns]
    
    # Return the selected features
    return data[top_features]

# Feature Engineering with Selected Features
def feature_engineering(data, selected_features):
    """
    Perform feature engineering on the dataset using selected features.

    Args:
        data (pd.DataFrame): Raw data.
        selected_features (pd.DataFrame): DataFrame containing only selected features.

    Returns:
        tuple: Processed features, labels, scaler, and polynomial transformer.
    """
    try:
        # Difference Features
        # Assuming selected_features already contains necessary features separated for home and away
        # If home and away features are combined, adjust accordingly

        # For example, if selected_features include both home and away stats with suffixes:
        home_features = selected_features.filter(regex='^[^\.]+$').copy()
        away_features = selected_features.filter(regex='^[^\.]+\.[^\.]+$').copy()
        away_features.columns = [col.replace('.1', '') for col in away_features.columns]

        # Ensure both dataframes have the same columns after renaming
        common_cols = home_features.columns.intersection(away_features.columns)
        home_features = home_features[common_cols]
        away_features = away_features[common_cols]

        # Difference Features
        diff_features = home_features - away_features
        diff_features = diff_features.add_suffix('_diff')

        # Days Rest Features
        diff_features['Days_Rest_Home_diff'] = data['Days-Rest-Home'] - data['Days-Rest-Away']

        # Rolling Averages for Points (PTS)
        # Ensure data is sorted by date to calculate rolling averages correctly
        data_sorted = data.sort_values(by='Date')  # Adjust if 'Date.1' is relevant for away team
        data_sorted['Last_5_Games_Avg_PTS_Home'] = data_sorted.groupby('TEAM_NAME')['PTS'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        data_sorted['Last_5_Games_Avg_PTS_Away'] = data_sorted.groupby('TEAM_NAME.1')['PTS'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        diff_features['Last_5_Games_Avg_PTS_diff'] = data_sorted['Last_5_Games_Avg_PTS_Home'] - data_sorted['Last_5_Games_Avg_PTS_Away']

        # Incorporate External Data (e.g., Injuries)

        # Additional Feature Engineering (Placeholder for future enhancements)
        # Example: Incorporate travel distance, player stats, etc.

        # Handle Missing Values
        diff_features.fillna(diff_features.mean(), inplace=True)

        # Polynomial Features
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(diff_features)

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(poly_features)

        # Labels
        y = data['Home-Team-Win'].values

        print("Feature engineering with selected features completed successfully.")
        return X_scaled, y, scaler, poly

    except Exception as e:
        print(f"Error during feature engineering: {e}")
        raise

# Example Feature Importance List
# Replace this with your actual feature importance list
# Format: [('feature_name1', importance_score1), ('feature_name2', importance_score2), ...]
important_features = [
    ('W_PCT', 99.0),
    ('PLUS_MINUS_RANK.1', 83.0),
    ('PLUS_MINUS_RANK', 33.0),
    ('W_PCT.1', 26.0),
    ('Days-Rest-Away', 22.0),
    ('W_RANK', 21.0),
    ('_PCT.1', 21.0),
    ('W_PCT Days-Rest-Home', 20.0),  # Interaction feature (to be created)
    ('GP', 19.0),
    ('PLUS_MINUS', 19.0),
    # Add more features as necessary
]

# Hyperparameter Optimization with Optuna for XGBoost
def optimize_hyperparameters(X, y):
    """
    Optimize hyperparameters for XGBoost using Optuna.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        model = xgb.XGBClassifier(**param)
        score = cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()
        return score

    study = optuna.create_study(direction='maximize', study_name="XGBoost_Hyperparameter_Optimization")
    study.optimize(objective, n_trials=50, timeout=600)  # 50 trials or 10 minutes
    print("Best hyperparameters found by Optuna:")
    print(study.best_params)
    return study.best_params

# Model Building: Stacking Ensemble
def build_stacking_model(best_xgb_params):
    """
    Build a stacking ensemble model.

    Args:
        best_xgb_params (dict): Best hyperparameters for XGBoost.

    Returns:
        StackingClassifier: Configured stacking ensemble.
    """
    try:
        # Base Estimators
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        xgb_clf = xgb.XGBClassifier(**best_xgb_params, use_label_encoder=False, eval_metric='logloss')

        estimators = [
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb_clf)
        ]

        # Final Estimator
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)

        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            passthrough=True,
            n_jobs=-1
        )

        print("Stacking model built successfully.")
        return stacking_clf
    except Exception as e:
        print(f"Error building stacking model: {e}")
        raise

# Revised SHAP Analysis Function using KernelExplainer
def shap_analysis(model, X_train, X_val, feature_names):
    """
    Perform SHAP analysis for model interpretability using KernelExplainer.

    Args:
        model: Trained stacking ensemble model.
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        feature_names (list): Names of the features.

    Returns:
        None
    """
    try:
        # Use a small subset of training data as background for faster computation
        background_size = min(100, X_train.shape[0])  # Adjust based on dataset size
        background = X_train[np.random.choice(X_train.shape[0], background_size, replace=False)]
        
        # Define a prediction function that returns probabilities for the positive class
        def predict_proba(X):
            return model.predict_proba(X)[:, 1]
        
        # Initialize SHAP KernelExplainer
        explainer = shap.KernelExplainer(predict_proba, background)
        
        # Compute SHAP values for the validation set
        shap_values = explainer.shap_values(X_val, nsamples=100)  # Adjust nsamples for speed vs. accuracy
        
        # Summary Plot for the positive class
        shap.summary_plot(shap_values, X_val, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png')
        plt.close()
        print("SHAP summary plot saved as 'shap_summary_plot.png'.")
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        raise

# Main Execution
if __name__ == "__main__":
    try:
        # Timestamp for logging and saving models
        current_time = str(int(time.time()))

        # Load Data
        dataset = "dataset_2012-25_new"  # Ensure this is the correct dataset name
        db_path = "../../Data/dataset.sqlite"  # Update path as necessary
        data = load_data(db_path, dataset)

        # Feature Selection
        X_selected = select_important_features(data, important_features, top_n=20)

        # Feature Engineering with Selected Features
        X, y, scaler, poly = feature_engineering(data, X_selected)

        # Feature Names after Polynomial Features
        feature_names = poly.get_feature_names_out()

        # Split Data - Time-Based Split to Prevent Leakage
        split_index = int(0.8 * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        print(f"Data split into training ({len(X_train)}) and validation ({len(X_val)}) sets.")

        # Hyperparameter Optimization for XGBoost
        print("Starting hyperparameter optimization for XGBoost...")
        best_params = optimize_hyperparameters(X_train, y_train)

        # Build Stacking Ensemble with Optimized XGBoost
        stacking_model = build_stacking_model(best_params)

        # Train Stacking Model
        print("Training stacking ensemble model...")
        stacking_model.fit(X_train, y_train)
        print("Model training completed.")

        # Predictions
        y_pred_proba = stacking_model.predict_proba(X_val)[:,1]
        y_pred_classes = stacking_model.predict(X_val)

        # Evaluation
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred_classes))
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        print(f"ROC-AUC: {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Loss', 'Predicted Win'], 
                    yticklabels=['Actual Loss', 'Actual Win'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("Confusion matrix saved as 'confusion_matrix.png'.")

        # SHAP Analysis for Interpretability
        print("Performing SHAP analysis...")
        shap_analysis(stacking_model, X_train, X_val, feature_names)

        # Save the Model and Scalers
        model_path = f'../../Models/Stacking-Model-{current_time}.joblib'
        scaler_path = f'../../Models/Scaler-{current_time}.joblib'
        poly_path = f'../../Models/PolyFeatures-{current_time}.joblib'

        joblib.dump(stacking_model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(poly, poly_path)
        print(f"Models and transformers saved successfully at '{model_path}', '{scaler_path}', and '{poly_path}'.")

        print('Model training and evaluation completed successfully.')

    except Exception as e:
        print(f"An error occurred during model execution: {e}")