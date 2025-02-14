import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

def clean_and_clip_features(df):
    """Clean and clip feature values to prevent infinity and extreme values"""
    print("Cleaning and clipping features...")
    
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # For each column, clip values to within 5 standard deviations of the mean
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:  # Only clip if standard deviation is positive
                lower_bound = mean - 5 * std
                upper_bound = mean + 5 * std
                df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def create_advanced_features(df):
    """Create advanced features with proper error handling"""
    print("Creating advanced features...")
    
    # 1. Moving averages for PLUS_MINUS related features
    window_sizes = [3, 5, 10]
    for window in window_sizes:
        # Home team moving averages
        df[f'PLUS_MINUS_MA_{window}'] = df.groupby('TEAM_NAME')['PLUS_MINUS'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().fillna(0))
        
        # Away team moving averages
        df[f'PLUS_MINUS_MA_{window}.1'] = df.groupby('TEAM_NAME.1')['PLUS_MINUS.1'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().fillna(0))
    
    # 2. Win percentage momentum (rate of change)
    df['W_PCT_MOMENTUM'] = df.groupby('TEAM_NAME')['W_PCT'].transform(
        lambda x: x.pct_change(periods=5).fillna(0).replace([np.inf, -np.inf], 0))
    df['W_PCT_MOMENTUM.1'] = df.groupby('TEAM_NAME.1')['W_PCT.1'].transform(
        lambda x: x.pct_change(periods=5).fillna(0).replace([np.inf, -np.inf], 0))
    
    # 3. Rest day impact features
    df['REST_ADVANTAGE'] = (df['Days-Rest-Home'] - df['Days-Rest-Away']).clip(-5, 5)
    df['REST_IMPACT_HOME'] = df['Days-Rest-Home'].map(lambda x: 1 if x >= 2 else 0)
    df['REST_IMPACT_AWAY'] = df['Days-Rest-Away'].map(lambda x: 1 if x >= 2 else 0)
    
    # 4. Recent form features (with bounds)
    df['RECENT_PLUS_MINUS_HOME'] = df.groupby('TEAM_NAME')['PLUS_MINUS'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().fillna(0).clip(-30, 30))
    df['RECENT_PLUS_MINUS_AWAY'] = df.groupby('TEAM_NAME.1')['PLUS_MINUS.1'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().fillna(0).clip(-30, 30))
    
    # 5. Win streak features (with maximum cap)
    def get_streak(group):
        streak = 0
        streaks = []
        for win in group:
            if win:
                streak = min(streak + 1, 15)  # Cap at 15
            else:
                streak = 0
            streaks.append(streak)
        return pd.Series(streaks)

    df['WIN_STREAK_HOME'] = df.groupby('TEAM_NAME')['W_PCT'].transform(
        lambda x: get_streak(x > x.shift(1)))
    df['WIN_STREAK_AWAY'] = df.groupby('TEAM_NAME.1')['W_PCT.1'].transform(
        lambda x: get_streak(x > x.shift(1)))
    
    # 6. Interaction features (with bounds)
    df['W_PCT_DIFF'] = (df['W_PCT'] - df['W_PCT.1']).clip(-1, 1)
    df['PLUS_MINUS_DIFF'] = (df['PLUS_MINUS'] - df['PLUS_MINUS.1']).clip(-50, 50)
    df['RANK_DIFF'] = (df['W_PCT_RANK'] - df['W_PCT_RANK.1']).clip(-29, 29)
    
    return df

def create_enhanced_h2h_features(df):
    """Create head-to-head features with proper error handling"""
    print("Creating enhanced head-to-head features...")
    h2h_stats = {}
    
    for _, row in df.iterrows():
        home_team = row['TEAM_NAME']
        away_team = row['TEAM_NAME.1']
        key = f"{home_team}_{away_team}"
        
        if key not in h2h_stats:
            h2h_stats[key] = []
            
        try:
            plus_minus = float(row['PLUS_MINUS'])
            if np.isfinite(plus_minus):  # Only add if value is finite
                h2h_stats[key].append({
                    'win': float(row['Home-Team-Win']),
                    'plus_minus': plus_minus,
                    'date': row['Date']
                })
        except (ValueError, TypeError):
            continue
    
    # Calculate h2h metrics with bounds
    df['H2H_WIN_RATE'] = df.apply(
        lambda x: np.mean([g['win'] for g in h2h_stats.get(f"{x['TEAM_NAME']}_{x['TEAM_NAME.1']}", [])][-10:] or [0.5]),
        axis=1
    )
    
    df['H2H_PLUS_MINUS_AVG'] = df.apply(
        lambda x: np.clip(
            np.mean([g['plus_minus'] for g in h2h_stats.get(f"{x['TEAM_NAME']}_{x['TEAM_NAME.1']}", [])][-10:] or [0]),
            -30, 30
        ),
        axis=1
    )
    
    return df

def load_and_preprocess_data(dataset_name, db_path):
    print("Loading data...")
    con = sqlite3.connect(db_path)
    data = pd.read_sql_query(f'select * from "{dataset_name}"', con, index_col="index")
    con.close()
    
    print("\nInitial data shape:", data.shape)
    
    # Convert date columns to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date.1'] = pd.to_datetime(data['Date.1'])
    
    # Add advanced features
    data = create_advanced_features(data)
    data = create_enhanced_h2h_features(data)
    
    # Target variable
    y = data['Home-Team-Win'].astype(float)
    
    # Drop non-feature columns
    columns_to_drop = ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 
                      'Date.1', 'OU-Cover', 'OU']
    X = data.drop(columns_to_drop, axis=1)
    
    # Convert features to numeric and clean
    print("\nConverting features to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Clean and clip features
    X = clean_and_clip_features(X)
    
    # Handle missing values
    print("\nHandling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    # Final verification
    assert not np.any(np.isnan(X.values)), "NaN values found in final dataset"
    assert not np.any(np.isinf(X.values)), "Infinite values found in final dataset"
    
    print("\nFinal preprocessed data shape:", X.shape)
    return X, y

def objective(trial, X, y):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        
        # Verify no NaN values before SMOTE
        assert not X_train_scaled.isnull().values.any(), "NaN values found before SMOTE"
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_balanced, y_train_balanced)
        
        y_pred = model.predict_proba(X_val_scaled)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)
    
    return np.mean(scores)

def main():
    print("Starting data loading and preprocessing...")
    X, y = load_and_preprocess_data("dataset_2012-25_new", "../../Data/dataset.sqlite")
    
    print("\nStarting hyperparameter optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    
    print("\nBest hyperparameters:", study.best_params)
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    
    print("\nTraining final model...")
    # Use the last 20% of the data for testing (maintaining temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Apply SMOTE to balance the training data
    print("Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Train final model
    print("Training final model with best parameters...")
    final_model = xgb.XGBClassifier(**best_params, random_state=42)
    final_model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, 
                                                             average='binary')
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print("\nFinal Model Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC-ROC: {auc_roc:.3f}")
    
    # Create feature importance visualization
    print("\nCreating feature importance plot...")
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('../../Models/feature_importance.png')
    plt.close()
    
    # Save the final model
    print("\nSaving model...")
    final_model.save_model('../../Models/XGBoost_final.json')
    
    # Save feature importances to CSV
    importance_df.to_csv('../../Models/feature_importances.csv', index=False)
    
    print("\nProcess completed successfully!")
    
    # Return the best score achieved during optimization
    return study.best_value

if __name__ == "__main__":
    try:
        best_score = main()
        print(f"\nBest optimization score: {best_score:.3f}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise