import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# check for GPU availability
def is_gpu_available():
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        return device_count > 0
    except Exception:
        return False

# Determine GPU or CPU
use_gpu = is_gpu_available()
if use_gpu:
    tree_method = "gpu_hist"
    device_str = "cuda"  #
    print("Training on GPU (CUDA available).")
else:
    tree_method = "hist"
    print("Training on CPU (CUDA not available).")

dataset = "dataset_2012-25_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True)

data = data.values.astype(float)
acc_results = []

for x in tqdm(range(300)):
    x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=0.1)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)

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
    model = xgb.train(param, train, epochs)
    predictions = model.predict(test)
    
    # Get predicted class labels
    y_pred = [np.argmax(pred) for pred in predictions]

    acc = round(accuracy_score(y_test, y_pred) * 100, 1)
    print(f"Iteration {x+1}: {acc}% accuracy")
    acc_results.append(acc)
    
    # Save the model if it yields the best accuracy so far
    if acc == max(acc_results):
        model.save_model(f'../../Models/XGBoost_{acc}%_ML-4.json')