import pandas as pd
import os
path_to_data = "./../datasets/loan/train.csv"
dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(dir,path_to_data))
print(df.head())

#preprocessing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df:
    if df[col].dtype == 'object':
        le.fit(df[col].values)
        df[col] = le.fit_transform(df[col])

print(df.head())

#preprocessing NaN
col = list(df.columns)