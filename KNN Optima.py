import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, f1_score, roc_auc_score, roc_curve
import kagglehub

# Memuat dataset
path = kagglehub.dataset_download("erdemtaha/cancer-data")
print("Dataset berhasil diunduh di:", path)

import os
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_path = os.path.join(path, file)
        print("File dataset ditemukan:", file)

data = pd.read_csv(csv_path)
# data = pd.read_csv('D:\Documents\BERKAS\PROJECT\Artifical Intelegence\Assignment 1\Breast_Cancer\Cancer_Data.csv') 
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Membuat Grafik persebaran Kanker Ganas dan Jinak
M = data[data.diagnosis == "M"] #Diagnosis transfers all values of M to M data
B = data[data.diagnosis == "B"] #Diagnosis transfers all values of B to B data
plt.scatter(M.radius_mean,M.texture_mean, color = "red", label = "Malignant", alpha = 0.3)
plt.scatter(B.radius_mean,B.texture_mean, color = "green", label = "Benign", alpha = 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# Mengubah data teks menjadi binner: M -> 1, B -> 0
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Min-Max normalization
X = (X - X.min()) / (X.max() - X.min())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Mencari nilai K optimal dengan GridSearchCV
param_grid = {'n_neighbors': list(range(1, 21))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Mengambil best K terbaik
best_k = grid.best_params_['n_neighbors']
print(f"Best k value: {best_k}")


# Ambil skor rata-rata untuk setiap K
k_values = param_grid['n_neighbors']
mean_scores = grid.cv_results_['mean_test_score']

plt.plot(k_values, mean_scores, marker='o', color='blue')
plt.axvline(best_k, color='red', linestyle='--', label=f"Best k = {best_k}")
plt.xlabel("k values")
plt.ylabel("accuracy vs k")
plt.show()

# KNN dengan K optimal
knn_optimal = KNeighborsClassifier(n_neighbors=best_k)
knn_optimal.fit(X_train, y_train)
prediction = knn_optimal.predict(X_test)
proba = knn_optimal.predict_proba(X_test)[:, 1]
print("Accuracy Score")
print("------------------------")
print("Acc Score {} ".format(knn_optimal.score(X_test,y_test)))
print("------------------------\n\n")

# Hitung metrik
accuracy = accuracy_score(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, proba)

# Menampilkan hasil KNN
print("\nEvaluation Metrics for K Optimal")
print("--------------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, prediction.reshape(-1))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title(f"Confusion Matrix - KNN (k={best_k})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
correct_predictions = np.trace(cm)
total_predictions = np.sum(cm)
incorrect_predictions = total_predictions - correct_predictions
plt.show()

# Resul Prediksi
print("Result Evaluation")
print("------------------------")
print(f'Sum True Prediction: {correct_predictions}\n')
print(f'Sum False Prediction: {incorrect_predictions}\n')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN")
plt.legend()
plt.show()