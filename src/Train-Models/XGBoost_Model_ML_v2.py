import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. Check for GPU availability ---
def is_gpu_available():
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        return device_count > 0
    except Exception:
        return False

use_gpu = is_gpu_available()
if use_gpu:
    tree_method = "gpu_hist"
    device_str = "cuda"
    print("Training on GPU (CUDA available).")
else:
    tree_method = "hist"
    print("Training on CPU (CUDA not available).")

# --- 2. Load and preprocess the data ---
dataset = "dataset_2012-25_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
# Read data into a DataFrame and keep column names
data_df = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

# Separate target and drop unused columns
margin = data_df['Home-Team-Win']
data_df.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date',
              'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1, inplace=True)

# Save the remaining feature names
feature_names = list(data_df.columns)
print("Features used for baseline:", feature_names)

# Convert data to a NumPy array (we’ll use the DataFrame later for subsetting)
data_np = data_df.values.astype(float)

# --- 3. Train a baseline XGBoost model on all features ---
x_train, x_test, y_train, y_test = train_test_split(data_np, margin, test_size=0.1, random_state=42)

dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)

param = {
    'max_depth': 3,
    'eta': 0.01,
    'objective': 'multi:softprob',
    'num_class': 2,
    'tree_method': tree_method,
}
if use_gpu:
    param['device'] = device_str

epochs = 1000
print("Training baseline model on all features...")
baseline_model = xgb.train(param, dtrain, epochs)

# --- 4. Use SHAP to find the top 5 features ---
print("Computing SHAP values...")
explainer = shap.TreeExplainer(baseline_model)
# For multiclass, shap_values is returned as a list (one array per class)
shap_values = explainer.shap_values(x_train)

if isinstance(shap_values, list):
    # Average the absolute SHAP values over classes and samples, then convert to a list
    shap_importance = np.mean(np.abs(np.array(shap_values)), axis=(0, 1)).tolist()
else:
    shap_importance = np.mean(np.abs(shap_values), axis=0).tolist()

# Create a dictionary mapping each feature to its importance.
importance_dict = dict(zip(feature_names, shap_importance))

# Helper function to convert an importance value to a scalar.
def to_scalar(val):
    if isinstance(val, list):
        if len(val) == 1:
            return float(val[0])
        else:
            return float(np.mean(val))
    else:
        return float(val)

# Sort features by importance (highest first) and take the top 5.
sorted_features = sorted(importance_dict.items(), key=lambda x: to_scalar(x[1]), reverse=True)
top_features = [f[0] for f in sorted_features[:5]]
print("Top 5 features identified by SHAP:", top_features)

# --- 5. Create an Optuna study to tune hyperparameters using only the top 5 features ---
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.001, 0.1, log=True),
        "objective": "multi:softprob",
        "num_class": 2,
        "tree_method": tree_method,
    }
    if use_gpu:
        params["device"] = device_str

    # Subset data to only the top 5 features
    X_top = data_df[top_features].values.astype(float)
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_top, margin, test_size=0.1, random_state=42
    )
    dtrain_sub = xgb.DMatrix(X_train_sub, label=y_train_sub, feature_names=top_features)
    dval_sub = xgb.DMatrix(X_val_sub, label=y_val_sub, feature_names=top_features)

    bst = xgb.train(params, dtrain_sub, num_boost_round=1000,
                    evals=[(dval_sub, "eval")],
                    early_stopping_rounds=50,
                    verbose_eval=False)
    # Use best_ntree_limit if available, else predict normally.
    if hasattr(bst, 'best_ntree_limit'):
        preds = bst.predict(dval_sub, ntree_limit=bst.best_ntree_limit)
    else:
        preds = bst.predict(dval_sub)
    y_pred = [np.argmax(pred) for pred in preds]
    acc = accuracy_score(y_val_sub, y_pred)
    # Return negative accuracy so that higher accuracy is better.
    return -acc

print("Starting Optuna study for hyperparameter tuning on top features...")
study = optuna.create_study()
study.optimize(objective, n_trials=50)

print("Best trial:")
best_trial = study.best_trial
print(f"  Accuracy: {-best_trial.value}")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Prepare best parameters for final training.
best_params = best_trial.params.copy()
best_params.update({
    "objective": "multi:softprob",
    "num_class": 2,
    "tree_method": tree_method,
})
if use_gpu:
    best_params["device"] = device_str

# --- 6. Final training using the best parameters on the top 5 features, with TQDM ---
X_top = data_df[top_features].values.astype(float)
best_acc = 0
acc_results = []
for i in tqdm(range(300), desc="Final Training Iterations"):
    # Vary the random state so each iteration uses a different train/test split.
    X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
        X_top, margin, test_size=0.1, random_state=i
    )
    dtrain_top = xgb.DMatrix(X_train_top, label=y_train_top, feature_names=top_features)
    dtest_top = xgb.DMatrix(X_test_top, label=y_test_top, feature_names=top_features)

    model = xgb.train(best_params, dtrain_top, num_boost_round=1000,
                      evals=[(dtest_top, "eval")],
                      early_stopping_rounds=50,
                      verbose_eval=False)
    if hasattr(model, 'best_ntree_limit'):
        preds = model.predict(dtest_top, ntree_limit=model.best_ntree_limit)
    else:
        preds = model.predict(dtest_top)
    y_pred = [np.argmax(pred) for pred in preds]
    acc = accuracy_score(y_test_top, y_pred)
    acc_results.append(acc)
    if acc > best_acc:
        best_acc = acc
        model.save_model(f'../../Models/XGBoost_top5_features_{acc*100:.1f}%.json')
        print(f"\nIteration {i+1}: New best accuracy: {acc*100:.1f}%")

print(f"Final best accuracy over 300 iterations: {best_acc*100:.1f}%")