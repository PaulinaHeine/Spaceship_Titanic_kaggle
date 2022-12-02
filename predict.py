import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Read files

train = pd.read_csv("/Users/paulinaheine/Codes/Spaceship_Titanic/spaceship-titanic_data/train.csv")
test = pd.read_csv("/Users/paulinaheine/Codes/Spaceship_Titanic/spaceship-titanic_data/test.csv")
sample = pd.read_csv("/Users/paulinaheine/Codes/Spaceship_Titanic/spaceship-titanic_data/sample_submission.csv")

# Prepare the Data

# Percentages of missing values
missing = round(100 * train.isnull().sum() / train.shape[0], 2)
missing_df = pd.DataFrame({'% of missing values':missing})
print(missing_df)
# show missings
# msno.bar(train, figsize=[8, 8], fontsize=10)
# plt.show()


#Cabin
train["Deck"] = list(train['Cabin'].str.split('/', expand = True)[0].copy())
train["Deckside"] = list(train['Cabin'].str.split('/', expand = True)[2].copy())
#train["CabinNumber"] = (train['Cabin'].str.split('/', expand = True)[1]).copy()

test["Deck"] = list(test['Cabin'].str.split('/', expand = True)[0].copy())
test["Deckside"] = list(test['Cabin'].str.split('/', expand = True)[2].copy())
#test["CabinNumber"] = (test['Cabin'].str.split('/', expand = True)[1]).copy()

# one hot encoding
train = pd.get_dummies(train, columns=["HomePlanet", "CryoSleep", "Destination", "Deck", "Deckside"])
test = pd.get_dummies(test, columns=["HomePlanet", "CryoSleep", "Destination", "Deck", "Deckside"])

# drop nans

train = train.drop(["Name", "Cabin"], axis=1)
test = test.drop(["Name", "Cabin"], axis=1)

for i in range(len(train.columns) - 1):
    train.iloc[:, [i]] = train.iloc[:, [i]].fillna(train.iloc[:, [i]].mean())

for i in range(len(test.columns)):
    test.iloc[:, [i]] = test.iloc[:, [i]].fillna(test.iloc[:, [i]].mean())

y = train["Transported"]
X = train.drop(["Transported"], axis=1)

# Prepare learning

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train_h, X_test_rest, y_train_h, y_test_rest = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# insert the best values
print("start")
gbc = GradientBoostingClassifier(
    loss="exponential",
    max_depth=3,
    n_estimators=375,
    learning_rate=0.08,
    random_state=42)

gbc = gbc.fit(X_test_rest, y_test_rest)

print(gbc.score(X_test_rest, y_test_rest))

for fn, fi in sorted(zip(X.columns, gbc.feature_importances_), key=lambda xx: xx[1], reverse=True):
    print(f"{fn}: {fi:.3f}")

y_pred = gbc.predict(test)

df_pred = pd.DataFrame(y_pred, columns=['Transported'])
df_pred["PassengerId"] = sample["PassengerId"]
df_pred = df_pred[['PassengerId', 'Transported']]

df_pred.to_csv("/Users/paulinaheine/Codes/Spaceship_Titanic/spaceship-titanic_data/prediction.csv", index=False)
