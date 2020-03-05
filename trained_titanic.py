from sklearn.externals import joblib
from os import system
import re

#import pandas as pd

columns= ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_test = [[0,0,0,0,0,0,0]]

def formulario():
    system("clear")
    print("Calcula tus posibilidades de sobrevivir en el titanic gracias a Machine Learning!")
    print("\n\n En que clase habrias viajado? (1, 2, 3)")
    X_test[0][0] = int(input())
    system("clear")

    print("\n\n Eres hombre o mujer? (h, m)")
    choice = input()
    if choice.lower().strip() =="h":
        X_test[0][1]=0
    elif choice.lower().strip() == "m":
        X_test[0][1]=1
    else:
        print("ups, no has introducido un valor correcto")
        x = input()
    
    system("clear")
    print("\n\n Que edad tienes?")
    X_test[0][2] = float(input())
    system("clear")

    print("\n\n Cuantos herman@s tienes?")
    X_test[0][3] = int(input())
    system("clear")
    
    print("\n\n Con cuantos hijos habrias viajado? sumale el esposo o la esposa")
    X_test[0][4] = int(input())
    system("clear")

    print("\n\n Cuanto habrias pagado por viajar en el Titanic? los precios variaban de 8 a 150$")
    X_test[0][5] = float(input())
    system("clear")

    print("\n\n Donde te subiste al barco? C = Cherbourg; Q = Queenstown; S = Southampton")
    choice = input()
    system("clear")
    Embarked = {"S":0, "C":1, "Q":2}
    choice = choice.upper().strip()
    X_test[0][6] = Embarked[choice]

formulario()

decision_tree = joblib.load('saved_model.pkl')  
prediction = decision_tree.predict_proba(X_test)

print("Habrias tenido una posibilidad de sobrevivir del: ",prediction[0][1]*100, "%")
