import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("kidney_disease.csv")

X = df.drop("classification", axis=1)
Y = df["classification"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,random_state=10000)

print(f"Training set size: {X_train.shape} Testing set size: {X_test.shape}")

#We shouldnt train and test models on the same data because the model can start to memorize the examples from training.
#this means that the model would do poorly when needed to predict data.
#the purpose of the testing set is so that we can see if the model can predict appropriately to new data
#that it hasnt seen. It shows us how well the model performs.

