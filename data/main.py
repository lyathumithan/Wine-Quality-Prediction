import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("data/Winequality-red.csv", sep=";")

print(data.head())
print(data.shape)
print(data.columns)
print(data.isnull().sum())
print(data["quality"].value_counts().sort_index())

data["quality"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Wine Quality")
plt.ylabel("Number of Wines")
plt.title("Distribution of Wine Quality")

plt.show()
print(data.groupby("quality")["alcohol"].mean())
correlation = data.corr()["quality"].sort_values(ascending=False)

print(correlation)
X = data.drop("quality", axis=1)
y = data["quality"]

print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions[:10])
print(y_test.iloc[:10].values)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print("MAE:", mae)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)

print("R² Score:", r2)

plt.scatter(y_test, predictions)

plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted Wine Quality")

plt.show()

importance = model.feature_importances_

feature_importance = pd.Series(
    importance,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)

new_wine = [[
    7.4,    # fixed acidity
    0.70,   # volatile acidity
    0.00,   # citric acid
    1.9,    # residual sugar
    0.076,  # chlorides
    11.0,   # free sulfur dioxide
    34.0,   # total sulfur dioxide
    0.9978, # density
    3.51,   # pH
    0.56,   # sulphates
    9.4     # alcohol
]]

prediction = model.predict(new_wine)

print("Predicted Wine Quality:", prediction[0])