import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

#How to create the model of machine learning for the trained_titanic.py, you dont need to run this file
#since the model is saved on the .pkl file

data = pd.read_csv("titanic.csv")
data = data.set_index("PassengerId")
data.drop(["Name", "Ticket", "Cabin"], inplace = True, axis = 1)
data = data.dropna()

encoder_dict = {"Sex":{"male":0, "female":1}, "Embarked":{"S":0, "C":1, "Q":2}}

data = data.replace(encoder_dict)
print(data.columns)
X_data = data.drop(['Survived'], axis=1)
Y_data = data['Survived']
decision_tree = ExtraTreesClassifier()
decision_tree.fit(X_data, Y_data)

#X_test = [[1,1,1,1,0,899,0]]
#prediction = decision_tree.predict_proba(X_test)

joblib.dump(decision_tree, 'saved_model.pkl')
 