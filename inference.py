import pandas as pd
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