import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, jaccard_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


df = pd.read_csv("heart.csv")

scaler = StandardScaler()

columns = ["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall"]

df.boxplot(fontsize=6)

trtbps_data = np.array(df["trtbps"])
trtbps_column = np.sort(np.array(df["trtbps"]))
trtbps_median = np.median(trtbps_column)
trtbps_q1, trtbps_q3 = np.percentile(trtbps_data, 25), np.percentile(trtbps_data, 75)
trtbps_IQR = trtbps_q3 - trtbps_q1
trtbps_upper, trtbps_lower = trtbps_q3 + (1.5 * trtbps_IQR), trtbps_q1 - (1.5 * trtbps_IQR)
df = df[df["trtbps"] <= trtbps_upper]
df = df[df["trtbps"] >= trtbps_lower]

chol_data = np.array(df["chol"])
chol_column = np.sort(np.array(df["chol"]))
chol_median = np.median(chol_column)
chol_q1, chol_q3 = np.percentile(chol_data, 25), np.percentile(chol_data, 75)
chol_IQR = chol_q3 - chol_q1
chol_upper, chol_lower = chol_q3 + (1.5 * chol_IQR), chol_q1 - (1.5 * chol_IQR)
df = df[df["chol"] <= chol_upper]
df = df[df["chol"] >= chol_lower]

thalachh_data = np.array(df["thalachh"])
thalachh_column = np.sort(np.array(df["thalachh"]))
thalachh_median = np.median(thalachh_column)
thalachh_q1, thalachh_q3 = np.percentile(thalachh_data, 25), np.percentile(thalachh_data, 75)
thalachh_IQR = thalachh_q3 - thalachh_q1
thalachh_upper, thalachh_lower = thalachh_q3 + (1.5 * thalachh_IQR), thalachh_q1 - (1.5 * thalachh_IQR)
df = df[df["thalachh"] <= thalachh_upper]
df = df[df["thalachh"] >= thalachh_lower]

oldpeak_data = np.array(df["oldpeak"])
oldpeak_column = np.sort(np.array(df["oldpeak"]))
oldpeak_median = np.median(oldpeak_column)
oldpeak_q1, oldpeak_q3 = np.percentile(oldpeak_data, 25), np.percentile(oldpeak_data, 75)
oldpeak_IQR = oldpeak_q3 - oldpeak_q1
oldpeak_upper, oldpeak_lower = oldpeak_q3 + (1.5 * oldpeak_IQR), oldpeak_q1 - (1.5 * oldpeak_IQR)
df = df[df["oldpeak"] <= oldpeak_upper]
df = df[df["oldpeak"] >= oldpeak_lower]

X = df[columns]
Y = df["output"]

X = scaler.fit_transform(X)

X = np.asarray(X)
Y = np.asarray(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Best k = 4 or k = 8

for i in range(3, 11):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)

    print(f"K = {i}")
    print(classification_report(y_test, predicted, labels=[0, 1]))
    print(f"F1_Score: {f1_score(y_test, predicted)}")
    print(f"Jaccard: {jaccard_score(y_test, predicted)}")
    cm = confusion_matrix(y_test, predicted, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("\n\n")

