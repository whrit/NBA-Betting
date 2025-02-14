import sqlite3
import pandas as pd

dataset = "dataset_2012-25_new"
con = sqlite3.connect("../../Data/dataset.sqlite")

# Read the table into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM \"{dataset}\"", con)
con.close()

print(list(df.columns))