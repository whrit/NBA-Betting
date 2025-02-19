import sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna

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
    print("Training on GPU (CUDA available).")
else:
    print("Training on CPU (CUDA not available).")

dataset = "dataset_2012-25_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True)

data = data.values.astype(float)

def objective(trial):
    
    param = {
        'objective': 'multiclass',
        'num_class': 2,
        'metric': 'multi_logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }
    if use_gpu:
        param['device'] = 'gpu'
    else:
        param['device'] = 'cpu'
    
    x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=0.1)
    train_set = lgb.Dataset(x_train, label=y_train)
    valid_set = lgb.Dataset(x_test, label=y_test, reference=train_set)
    
    gbm = lgb.train(param,
                    train_set,
                    num_boost_round=1000,
                    valid_sets=[valid_set],
                    early_stopping_rounds=50,
                    verbose_eval=False)
    
    preds = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    y_pred = np.argmax(preds, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    # Optuna minimizes the objective so we return the negative accuracy
    return -accuracy

study = optuna.create_study()
study.optimize(objective, n_trials=50)

print("Best trial:")
best_trial = study.best_trial
print(f"  Accuracy: {-best_trial.value}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

best_params = best_trial.params.copy()
best_params.update({
    'objective': 'multiclass',
    'num_class': 2,
    'metric': 'multi_logloss'
})
if use_gpu:
    best_params['device'] = 'gpu'
else:
    best_params['device'] = 'cpu'

acc_results = []
for i in tqdm(range(300)):
    x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=0.1)
    train_set = lgb.Dataset(x_train, label=y_train)
    valid_set = lgb.Dataset(x_test, label=y_test, reference=train_set)
    
    gbm = lgb.train(best_params,
                    train_set,
                    num_boost_round=1000,
                    valid_sets=[valid_set],
                    early_stopping_rounds=50,
                    verbose_eval=False)
    
    preds = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    y_pred = np.argmax(preds, axis=1)
    acc = round(accuracy_score(y_test, y_pred) * 100, 1)
    print(f"Iteration {i+1}: {acc}% accuracy")
    acc_results.append(acc)
    
    # Save the model if it achieves the best accuracy so far
    if acc == max(acc_results):
        gbm.save_model(f'../../Models/LightGBM_{acc}%_ML-4.txt')