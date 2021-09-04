import pandas as pd
import os
import numpy as np
data_path = './../datasets/loan/train.csv'
dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(dir,data_path))
# print(df.head())
# print(df.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns.values:
    if df[col].dtype == 'object':
        le.fit(df[col].values)
        df[col] = le.transform(df[col])
#Null Value Imputation
rev_null=['Gender','Married','Dependents','Self_Employed','Credit_History','LoanAmount','Loan_Amount_Term']
df[rev_null]=df[rev_null].replace({np.nan:df['Gender'].mode(),
                                   np.nan:df['Married'].mode(),
                                   np.nan:df['Dependents'].mode(),
                                   np.nan:df['Self_Employed'].mode(),
                                   np.nan:df['Credit_History'].mode(),
                                   np.nan:df['LoanAmount'].mean(),
                                   np.nan:df['Loan_Amount_Term'].mean()})
print(df.head())
X = df.drop(columns=["Loan_ID","Loan_Status"]).values
Y = df['Loan_Status'].values

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape)
print(Y_train.shape)

from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train,Y_train)
dt_pred_train =dt.predict(X_train)
print(dt_pred_train.shape)
print(Y_train.shape)

from sklearn.metrics import f1_score
print('Training Set Evaluation F1 score', f1_score(Y_train,dt_pred_train))