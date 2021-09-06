import pandas as pd
import os
path_to_data = "./../datasets/loan/train.csv"
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_to_data))
# print(df.head())

#fFEATURE ENGINEERING- preprocessing data
#converting str values to numericals using LabelEncoder from Scikit
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df:
    if df[col].dtype == 'object':
        le.fit(df[col].values)
        df[col] = le.transform(df[col])
        df[col] = df[col].fillna(df[col].mode())
    else:
        df[col] = df[col].fillna(df[col].mean())
# print(df.head()) #should have str values mapped to integers
print(df.head())

#Split data into train and test
from sklearn.model_selection import train_test_split
X = df.drop(columns=["Loan_ID","Loan_Status"])
Y = df["Loan_Status"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=42)

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X,Y)
print(clf.feature_importances_)

#feature selection and Union for transformers
from sklearn.feature_selection import SelectFromModel
transformers = list()
transformers.append(("et",SelectFromModel(ExtraTreesClassifier(n_estimators=50),prefit=False)))

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
fu = FeatureUnion(transformers)
model = RandomForestClassifier(criterion='entropy',random_state=42)
#creating pipeline here
steps = list()
steps.append(('fu',fu))
steps.append(('m',model))
pipeline = Pipeline(steps=steps)
pipeline.fit(X_train,Y_train)
predicted_Y_train = pipeline.predict(X_train)
predicted_Y_test = pipeline.predict(X_test)

from sklearn.metrics import f1_score
print("Training Scores", f1_score(predicted_Y_train,Y_train))
print("Testing Scores",f1_score(predicted_Y_test,Y_test))