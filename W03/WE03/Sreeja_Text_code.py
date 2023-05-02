import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle


sreeja_winning_model = pickle.load(open('sreeja_winning_model.pkl', "rb"))

print("\n")
print("Prediction_model of lawnmower")
print("\n")
income = float(input("Enter the Income: "))
lot_size = float(input("Enter the lot size"))
df = pd.DataFrame({'Income': [income]},{'Lot_size':[lot_size]})
result = sreeja_winning_model.predict(df)
probability = sreeja_winning_model.predict_proba(df)
Ownership = ('there exists no ownership', 'there exists ownership')
print(f"\n Prediction model of the lawnmower is at {probability[0][1]:.4f}, therefore it have pointed out that {Ownership[result[0]]}.\n")
