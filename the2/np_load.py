import numpy as np
import pickle


"""
b= np.load("train_hists_8_128.npy", allow_pickle=True) # load
print(b)

for i,a in enumerate(b):
	print(a)
	if i == 2: break
    

def conf_matrix(preds, true_labels):
	TP = 0
	FP = 0
	TN = 0
	FN = 0


with open("kmeans_64.pkl", "rb") as f:
    model = pickle.load(f)

    print(model.cluster_centers_)

"""


from sklearn import metrics
from sklearn import preprocessing

true_labels = ["apple","orange","cherry","cherry","cherry","orange"]
preds=["apple","apple","cherry","cherry","cherry","cherry"]


le = preprocessing.LabelEncoder()
true_labels1 = le.fit_transform(np.asarray(true_labels))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("mapping: ", le_name_mapping)
print("encoded true labels: ", true_labels1)
preds1 = le.fit_transform(np.asarray(preds))
print("encoded true labels: ", preds1)




cm = metrics.confusion_matrix(true_labels1, preds1)
print(cm)