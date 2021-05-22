import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle


################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("wine_quality.csv")

# Split into X & y
y = df.pop("quality")
X = df.drop(columns=["quality"])

with open('clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X))