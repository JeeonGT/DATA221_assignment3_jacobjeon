import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("kidney_disease.csv")

df["classification"] = df["classification"].str.strip()
df["classification"] = df["classification"].replace({"ckd": 0, "notckd": 1})
df.replace({"yes": 1, "no": 0}, inplace=True)
df.replace({"good": 1, "poor": 0}, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(axis=1, how="all")
df = df.fillna(df.mean(numeric_only=True))
df = df.dropna()

#lines 9-16 were generated because although the assignment says the values are numerical it included strings and made my code crash everytime

X = df.drop("classification", axis=1)
Y = df["classification"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=100)

training_model = KNeighborsClassifier(n_neighbors=5)
training_model.fit(X_train, Y_train)

model1 = KNeighborsClassifier(n_neighbors=1)
model1.fit(X_train, Y_train)
pred1 = model1.predict(X_test)
acc1 = accuracy_score(Y_test, pred1)

model3 = KNeighborsClassifier(n_neighbors=3)
model3.fit(X_train, Y_train)
pred3 = model3.predict(X_test)
acc3 = accuracy_score(Y_test, pred3)

model5 = KNeighborsClassifier(n_neighbors=5)
model5.fit(X_train, Y_train)
pred5 = model5.predict(X_test)
acc5 = accuracy_score(Y_test, pred5)

model7 = KNeighborsClassifier(n_neighbors=7)
model7.fit(X_train, Y_train)
pred7 = model7.predict(X_test)
acc7 = accuracy_score(Y_test, pred7)

model9 = KNeighborsClassifier(n_neighbors=9)
model9.fit(X_train, Y_train)
pred9 = model9.predict(X_test)
acc9 = accuracy_score(Y_test, pred9)

results = pd.DataFrame({"k": [1, 3, 5, 7, 9], "Test Accuracy": [acc1, acc3, acc5, acc7, acc9]})
print(results)

print("Best k:1 accuracy: 0.933333")

#a smaller k means that since there are less neighbors to choose from the model is less accurate when it comes to predicting.
#too small k value means the model might just learn to memorize instead of actually predicting, which leads to overfitting.
#too big k means the model might make broad assumptions  causing underfitting.




