import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, f1_score, roc_auc_score, roc_curve
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data graph drawing
import kagglehub

path = kagglehub.dataset_download("erdemtaha/cancer-data")

print("Dataset berhasil diunduh di:", path)

import os
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_path = os.path.join(path, file)
        print("File dataset ditemukan:", file)

# data = pd.read_csv('D:\Documents\BERKAS\PROJECT\Artifical Intelegence\Assignment 1\Breast_Cancer\Cancer_Data.csv') 
# data = pd.read_csv('/kaggle/input/cancer-data/Cancer_Data.csv') 

data = pd.read_csv(csv_path)
data.info() # general information of the data
print(data["Unnamed: 32"]) # a column that is completely null
print(data["id"]) # an unnecessary value for the algorithm
data.tail()

data.drop(["id","Unnamed: 32"], axis = 1, inplace = True) #"axis = 1" indicates to delete the whole column |"inpalce = True" means replace master data, does not create a copy
data.info()

M = data[data.diagnosis == "M"] #Diagnosis transfers all values of M to M data
B = data[data.diagnosis == "B"] #Diagnosis transfers all values of B to B data

plt.scatter(M.radius_mean,M.texture_mean, color = "red", label = "Malignant", alpha = 0.3)
plt.scatter(B.radius_mean,B.texture_mean, color = "green", label = "Benign", alpha = 0.3)

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")

plt.legend()
plt.show()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis
x_data = data.drop(["diagnosis"],axis = 1)
print("X Data \n",x_data)
print("Y Data \n",y)

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

# X data info
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)

# X_train data info
print("X Train")
x_train

# y_train data info
print("Y Train")
y_train

print("X Test")
x_test

print("Y Test")
y_test

from sklearn.neighbors import KNeighborsClassifier
Score_list = []

for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    Score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),Score_list)
plt.xlabel("k values")
plt.ylabel("accuracy vs k")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())


knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
proba = knn.predict_proba(x_test)[:, 1] #Probalitas

print("Accuracy Score")
print("------------------------")
print("{} nn Acc Score {} ".format(3,knn.score(x_test,y_test)))
print("------------------------\n\n")

#seaborntable
cm = confusion_matrix(y_test, prediction.reshape(-1))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Prediction')
plt.ylabel('True Values')
plt.title('Classifcation Result')
correct_predictions = np.trace(cm)
total_predictions = np.sum(cm)
incorrect_predictions = total_predictions - correct_predictions
plt.show()

#Result
print("\n")
print("Result Evaluation")
print("------------------------")
print(f'Sum True Prediction: {correct_predictions}\n')
print(f'Sum False Prediction: {incorrect_predictions}\n')

# Hitung metrik tambahan
accuracy = accuracy_score(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))  # RMSE manual
f1 = f1_score(y_test, prediction)
auc = roc_auc_score(y_test, proba)

# Cetak hasil
print("\nAdditional Evaluation Metrics")
print("-----------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# 8. ROC Curve
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN")
plt.legend()
plt.show()
