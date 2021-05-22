import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("wine_quality.csv")

# Split into train and test sections
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=seed)

#################################
########## MODELLING ############
3#################################

# Fit a model on the train section
regr = RandomForestRegressor(max_depth=6, random_state=seed)
regr.fit(X_train, y_train)

# Save model in pickle
with open('clf.pickle', 'wb') as f:
    pickle.dump(regr, f)

# Report training set score
train_score = regr.score(X_train, y_train) * 100
print(train_score)
# Report test set score
test_score = regr.score(X_test, y_test) * 100
print(test_score)

# # Write scores to a file
# with open("metrics.txt", 'w') as outfile:
#         outfile.write("Training variance explained: %2.1f%%\n" % train_score)
#         outfile.write("Test variance explained: %2.1f%%\n" % test_score)


