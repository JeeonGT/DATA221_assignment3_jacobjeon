import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

Y_pred = training_model.predict(X_test)

confusionmatrix = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, pos_label=0, zero_division=0)
recall = recall_score(Y_test, Y_pred, pos_label=0, zero_division=0)
f1score = f1_score(Y_test, Y_pred, pos_label=0, zero_division=0)

print("Confusion Matrix (rows=true, cols=pred; labels=[0(CKD), 1(notckd)]):")
print(confusionmatrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1score)

#True positive: model predicts kidney disease in the patient and the patient has kidney disease
#True negative: model predicts not kidney disease while the patient truly has no kidney disease
#False positive: model predicts kidney disease while patient doesnt have kidney disease
#False negative: model predicts not kidney disease while patient truly has kidney disease.
#accuracy may not be enough because accuracy might be predicting the most common class and not giving true accurate responses.
#recall, because it measures how many actual cases the model is successfully predicting, thereby eliminating false negatives
