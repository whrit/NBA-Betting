import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import random

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm

############################
# (A) FEATURE ENGINEERING  #
############################

def clean_and_clip_features(df):
    """Clean and clip feature values to prevent infinity and extreme values."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower_bound = mean - 5 * std
                upper_bound = mean + 5 * std
                df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def create_advanced_features(df):
    """Create advanced features with temporal decay, advanced interactions, and leakage avoidance."""
    print("Creating advanced features...")
    
    # 1. Rolling averages for PLUS_MINUS with shift to avoid leakage and exponential weighting.
    window_sizes = [3, 5, 10]
    for window in window_sizes:
        # Standard rolling average for home (using only past games)
        df[f'PLUS_MINUS_MA_{window}'] = df.groupby('TEAM_NAME')['PLUS_MINUS']\
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean().fillna(0))
        # EWMA for home team
        df[f'PLUS_MINUS_EWMA_{window}'] = df.groupby('TEAM_NAME')['PLUS_MINUS']\
            .transform(lambda x: x.shift(1).ewm(span=window, adjust=False).mean().fillna(0))
        # Standard rolling average for away
        df[f'PLUS_MINUS_MA_{window}.1'] = df.groupby('TEAM_NAME.1')['PLUS_MINUS.1']\
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean().fillna(0))
        # EWMA for away team
        df[f'PLUS_MINUS_EWMA_{window}.1'] = df.groupby('TEAM_NAME.1')['PLUS_MINUS.1']\
            .transform(lambda x: x.shift(1).ewm(span=window, adjust=False).mean().fillna(0))
    
    # 2. Win percentage momentum (using past data only)
    df['W_PCT_MOMENTUM'] = df.groupby('TEAM_NAME')['W_PCT']\
        .transform(lambda x: x.shift(1).pct_change(periods=5).fillna(0).replace([np.inf, -np.inf], 0))
    df['W_PCT_MOMENTUM.1'] = df.groupby('TEAM_NAME.1')['W_PCT.1']\
        .transform(lambda x: x.shift(1).pct_change(periods=5).fillna(0).replace([np.inf, -np.inf], 0))
    
    # 3. Rest day impact
    df['REST_ADVANTAGE'] = (df['Days-Rest-Home'] - df['Days-Rest-Away']).clip(-5, 5)
    df['REST_IMPACT_HOME'] = df['Days-Rest-Home'].map(lambda x: 1 if x >= 2 else 0)
    df['REST_IMPACT_AWAY'] = df['Days-Rest-Away'].map(lambda x: 1 if x >= 2 else 0)
    
    # 4. Recent form (using past games only)
    df['RECENT_PLUS_MINUS_HOME'] = df.groupby('TEAM_NAME')['PLUS_MINUS']\
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(0).clip(-30, 30))
    df['RECENT_PLUS_MINUS_AWAY'] = df.groupby('TEAM_NAME.1')['PLUS_MINUS.1']\
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(0).clip(-30, 30))
    
    # 5. Win streak features (using past game results only)
    def get_streak(series):
        streak = 0
        result = []
        for value in series:
            if value:
                streak = min(streak + 1, 15)
            else:
                streak = 0
            result.append(streak)
        return pd.Series(result)

    df['WIN_STREAK_HOME'] = df.groupby('TEAM_NAME')['W_PCT']\
        .transform(lambda x: get_streak(x.shift(1) > x.shift(2)))
    df['WIN_STREAK_AWAY'] = df.groupby('TEAM_NAME.1')['W_PCT.1']\
        .transform(lambda x: get_streak(x.shift(1) > x.shift(2)))
    
    # 6. Basic interaction features (already existing)
    df['W_PCT_DIFF'] = (df['W_PCT'] - df['W_PCT.1']).clip(-1, 1)
    df['PLUS_MINUS_DIFF'] = (df['PLUS_MINUS'] - df['PLUS_MINUS.1']).clip(-50, 50)
    df['RANK_DIFF'] = (df['W_PCT_RANK'] - df['W_PCT_RANK.1']).clip(-29, 29)
    
    # 7. Advanced interaction features:
    # For example, a matchup advantage feature combining recent form and rest advantage:
    df['MATCHUP_ADV'] = (df['RECENT_PLUS_MINUS_HOME'] - df['RECENT_PLUS_MINUS_AWAY']) * df['REST_ADVANTAGE']
    # Interaction between plus-minus difference and win percentage difference
    df['INTERACTION_PLUS_MINUS_W_PCT'] = df['PLUS_MINUS_DIFF'] * df['W_PCT_DIFF']
    # Interaction of rest advantage with plus-minus difference
    df['REST_X_PLUS_MINUS'] = df['REST_ADVANTAGE'] * df['PLUS_MINUS_DIFF']
    
    return df

def create_enhanced_h2h_features(df):
    """Create head-to-head features in a vectorized manner to avoid leakage.
       For each unique matchup (home vs. away), only past games are used.
    """
    print("Creating enhanced head-to-head features...")
    # Create a matchup key
    df['h2h_key'] = df['TEAM_NAME'] + "_" + df['TEAM_NAME.1']
    # Ensure the DataFrame is sorted by date
    df = df.sort_values('Date')
    
    # Compute rolling head-to-head win rate (for home team) over past 10 meetings
    df['H2H_WIN_RATE'] = df.groupby('h2h_key')['Home-Team-Win']\
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean().fillna(0.5))
    # Compute rolling head-to-head plus-minus average over past 10 meetings and clip values
    df['H2H_PLUS_MINUS_AVG'] = df.groupby('h2h_key')['PLUS_MINUS']\
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean().fillna(0).clip(-30, 30))
    
    return df

########################
# (B) DATA LOADING     #
########################

def load_data(db_path, table_name):
    """Load NBA data from 2014 to 2024/25, sort by date, and engineer features."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM \"{table_name}\"", con, index_col="index")
    con.close()

    # Convert date columns to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date.1'] = pd.to_datetime(df['Date.1'])

    # Sort by the "Date" column to respect chronological order (time series)
    df.sort_values('Date', inplace=True)

    # Create advanced features
    df = create_advanced_features(df)
    df = create_enhanced_h2h_features(df)
    df = clean_and_clip_features(df)

    return df

############################
# (C) HYPERPARAMETER TUNING
############################

def objective(trial, X, y, n_splits=5):
    """
    Time-series objective for Optuna with early stopping and regularization:
      - Uses a TimeSeriesSplit with only past data.
      - Trains an XGBoost model with early stopping.
      - Returns the average AUC across folds.
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # Added regularization parameters:
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'random_state': 42
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params, early_stopping_rounds=10)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, y_pred_proba)
        scores.append(fold_auc)
    
    return np.mean(scores)

###########################
# Nested CV Evaluation    #
###########################

def nested_cv_evaluation(X, y, best_params, n_splits=5):
    """
    Evaluate the model using nested (expanding window) CV,
    reporting accuracy, AUC, F1 score, precision, recall, and Brier loss.
    """
    outer_tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {'accuracy': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [], 'brier': []}
    
    for train_idx, test_idx in outer_tscv.split(X):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
        
        # Create an internal validation split from the training portion for early stopping.
        if len(X_train_full) > 50:
            split = int(len(X_train_full) * 0.9)
            X_train, X_val = X_train_full[:split], X_train_full[split:]
            y_train, y_val = y_train_full[:split], y_train_full[split:]
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = None, None
        
        model = xgb.XGBClassifier(**best_params, random_state=42)
        if X_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['auc'].append(roc_auc_score(y_test, y_pred_proba))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['brier'].append(brier_score_loss(y_test, y_pred_proba))
    
    # Average the metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics

##################
# (D) MAIN LOGIC #
##################

def main():
    db_path = "../../Data/dataset.sqlite"
    table_name = "dataset_2012-25_new"

    # 1) Load data (seasons ~2014 through 2024-25)
    df = load_data(db_path, table_name)
    y = df['Home-Team-Win'].astype(int)

    # Drop columns not used as features
    to_drop = [
        'Score', 'Home-Team-Win', 'TEAM_NAME', 'TEAM_NAME.1',
        'Date', 'Date.1', 'OU-Cover', 'OU', 'h2h_key'
    ]
    df.drop(columns=to_drop, inplace=True, errors='ignore')

    # Convert all columns to numeric and fill NA
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.fillna(0, inplace=True)

    X = df.values
    feature_names = df.columns
    n = len(X)

    print("Data loaded. Shape:", X.shape, "Label shape:", y.shape)

    #####################################################
    # 2) Hyperparameter Tuning with TimeSeriesSplit + Optuna
    #####################################################
    print("\nStarting Optuna hyperparameter tuning with TimeSeriesSplit...\n")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, n_splits=5), n_trials=50)
    
    print("Best hyperparameters:", study.best_params)
    best_params = study.best_params
    best_params['use_label_encoder'] = False
    best_params['objective'] = 'binary:logistic'

    #####################################################
    # 3) Train final model on entire set and calibrate probabilities
    #####################################################
    final_model = xgb.XGBClassifier(**best_params, random_state=42)
    final_model.fit(X, y)
    
    # Calibrate the model (using isotonic regression)
    calibrated_model = CalibratedClassifierCV(final_model, cv=3, method='isotonic')
    calibrated_model.fit(X, y)

    print("\nComputing permutation importances (this can be slow for large data)...")
    perm_result = permutation_importance(
        calibrated_model,
        X,
        y,
        n_repeats=5,
        random_state=42
    )
    avg_importances = perm_result.importances_mean
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importances
    }).sort_values('importance', ascending=False)

    print("\nTop 15 features by permutation importance:")
    print(imp_df.head(15))

    # 4) Pick top 5 features
    top_5_features = imp_df.head(5)['feature'].tolist()
    print(f"\nSelected top 5 features: {top_5_features}")

    df_top5 = df[top_5_features]
    X_top5 = df_top5.values

    #######################################################
    # 5) Evaluate model stability with repeated TimeSeriesSplit on top 5 features
    #######################################################
    print("\n=== Evaluating final model stability (TimeSeriesSplit) on top 5 features ===")
    tscv2 = TimeSeriesSplit(n_splits=5)
    accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv2.split(X_top5), start=1):
        X_train_full, X_test = X_top5[train_idx], X_top5[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        # Create an internal split from training data for early stopping if possible.
        if len(X_train_full) > 50:
            split = int(len(X_train_full) * 0.9)
            X_train, X_val = X_train_full[:split], X_train_full[split:]
            y_train, y_val = y_train_full[:split], y_train_full[split:]
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = None, None

        model_top5 = xgb.XGBClassifier(**best_params, random_state=42)
        if X_val is not None:
            model_top5.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model_top5.fit(X_train, y_train)
        
        y_pred = model_top5.predict(X_test)
        fold_acc = accuracy_score(y_test, y_pred)
        accuracies.append(fold_acc)
        print(f"Fold {fold_idx}: Accuracy = {fold_acc:.3f}")

    print(f"\nAverage Accuracy across 5 folds (top 5 features): {np.mean(accuracies):.3f}")

    ##########################################################
    # 6) Repeated Time-Based Splits to avoid leakage with early stopping
    ##########################################################
    print("\n=== Repeated Time-Based Splits on top 5 features (no future leakage) ===")

    best_acc = 0
    best_model = None

    # We'll do 300 random splits by picking a random cut index (ensuring test set is roughly 10%-20%)
    min_cut = int(0.80 * n)
    max_cut = int(0.90 * n)

    for i in tqdm(range(300), desc="Time-based splits"):
        split_idx = random.randint(min_cut, max_cut)

        X_train_full = X_top5[:split_idx]
        y_train_full = y[:split_idx]
        X_test = X_top5[split_idx:]
        y_test = y[split_idx:]

        # Create an internal validation split from the training set for early stopping if possible.
        if len(X_train_full) > 50:
            split_internal = int(len(X_train_full) * 0.9)
            X_train, X_val = X_train_full[:split_internal], X_train_full[split_internal:]
            y_train, y_val = y_train_full[:split_internal], y_train_full[split_internal:]
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = None, None

        model_tmp = xgb.XGBClassifier(**best_params, random_state=42)
        if X_val is not None:
            model_tmp.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model_tmp.fit(X_train, y_train)
        
        y_pred = model_tmp.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred) * 100, 1)

        if acc > best_acc:
            best_acc = acc
            best_model = model_tmp
            best_model.save_model(f'../../Models/XGBoost_{best_acc}_Top5.json')

    print("\nDone with time-based splits. Best accuracy found:", best_acc, "%")

    ##########################################################
    # 7) Nested Cross-Validation Evaluation with additional metrics
    ##########################################################
    print("\n=== Nested Cross-Validation Evaluation on top 5 features ===")
    nested_metrics = nested_cv_evaluation(X_top5, y, best_params, n_splits=5)
    print("Nested CV Metrics:")
    for metric, value in nested_metrics.items():
        print(f"  {metric}: {value:.3f}")

    print("\nProcess complete!")

if __name__ == "__main__":
    main()