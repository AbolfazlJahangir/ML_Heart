import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
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

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

# Best solver = newton

model_lbfgs = LogisticRegression(C=1, solver="lbfgs")
model_lbfgs.fit(x_train, y_train)

predicted_lbfgs = model_lbfgs.predict(x_test)

print("lbfgs")
print(classification_report(y_test, predicted_lbfgs, labels=[0, 1]))
print(f"F1_Score: {f1_score(y_test, predicted_lbfgs)}")
print(f"Jaccard: {jaccard_score(y_test, predicted_lbfgs)}")
cm = confusion_matrix(y_test, predicted_lbfgs, labels=model_lbfgs.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_lbfgs.classes_)
disp.plot()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\n")


model_liblinear = LogisticRegression(C=1, solver="liblinear")
model_liblinear.fit(x_train, y_train)

predicted_liblinear = model_liblinear.predict(x_test)

print("liblinear")
print(classification_report(y_test, predicted_liblinear, labels=[0, 1]))
print(f"F1_Score: {f1_score(y_test, predicted_liblinear)}")
print(f"Jaccard: {jaccard_score(y_test, predicted_liblinear)}")
cm = confusion_matrix(y_test, predicted_liblinear, labels=model_liblinear.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_liblinear.classes_)
disp.plot()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\n")


model_newtoncg = LogisticRegression(C=1, solver="newton-cg")
model_newtoncg.fit(x_train, y_train)

predicted_newtoncg = model_newtoncg.predict(x_test)

print("newtoncg")
print(classification_report(y_test, predicted_newtoncg, labels=[0, 1]))
print(f"F1_Score: {f1_score(y_test, predicted_newtoncg)}")
print(f"Jaccard: {jaccard_score(y_test, predicted_newtoncg)}")
cm = confusion_matrix(y_test, predicted_newtoncg, labels=model_newtoncg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_newtoncg.classes_)
disp.plot()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\n")


model_newton_cholesky = LogisticRegression(C=1, solver="newton-cholesky")
model_newton_cholesky.fit(x_train, y_train)

predicted_newton_cholesky = model_newton_cholesky.predict(x_test)

print("newton_cholesky")
print(classification_report(y_test, predicted_newton_cholesky, labels=[0, 1]))
print(f"F1_Score: {f1_score(y_test, predicted_newton_cholesky)}")
print(f"Jaccard: {jaccard_score(y_test, predicted_newton_cholesky)}")
cm = confusion_matrix(y_test, predicted_newton_cholesky, labels=model_newton_cholesky.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_newton_cholesky.classes_)
disp.plot()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\n")


model_sag = LogisticRegression(C=1, solver="sag")
model_sag.fit(x_train, y_train)

predicted_sag = model_sag.predict(x_test)

print("sag")
print(classification_report(y_test, predicted_sag, labels=[0, 1]))
print(f"F1_Score: {f1_score(y_test, predicted_sag)}")
print(f"Jaccard: {jaccard_score(y_test, predicted_sag)}")
cm = confusion_matrix(y_test, predicted_sag, labels=model_sag.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_sag.classes_)
disp.plot()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\n")


model_saga = LogisticRegression(C=1, solver="saga")
model_saga.fit(x_train, y_train)

predicted_saga = model_saga.predict(x_test)

print("saga")
print(classification_report(y_test, predicted_saga, labels=[0, 1]))
print(f"F1_Score: {f1_score(y_test, predicted_saga)}")
print(f"Jaccard: {jaccard_score(y_test, predicted_saga)}")
cm = confusion_matrix(y_test, predicted_saga, labels=model_saga.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_saga.classes_)
disp.plot()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\n")
