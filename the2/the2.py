import cv2 
import os
import numpy as np
from sklearn.cluster import KMeans
import pickle


####################################
####################################
####################################
########## PARAMETERS ##############
####################################

dense = 1 # set dense = 1 for dense sift, dense = 0 o.w

n_clusters = 256 #k for k-means 
k=16 #k-NN's parameter k

## SIFT PARAMETERS
nfeatures = 0
nOctaveLayers = 3
contrastThreshold = 0.04
edgeThreshold = 10
sigma = 1.6

## DENSE SIFT PARAMETERS
step_size = 5

#dataset paths
train_path = "the2_data/train"
val_path = "the2_data/validation"
test_path = "the2_test/test"

test = 0 # set test 1 to run for the test set mode
show_confusion_matrix = 0 # set 1 to see the confusion matrix for validation set

####################################
####################################
####################################
####################################
####################################

#test output file
test_out_file = "test_results.txt"


#db file name
train_hist_file = "train_means_256_clusters_sift"




#GLOBAL VARS
train_features = dict() 	# {<img_name> : <descriptor>}
train_histograms = dict() # {<img_name> : <histogram>}
test_features = dict() 	# {<img_name> : <descriptor>}
test_histograms = dict() # {<img_name> : <histogram>}
all_descriptors = []
kmeans = None



def build_hists(mode):
	if mode == "train":
		for img, desc in train_features.items():
			preds = kmeans.predict(desc)
			hist, _ = np.histogram(preds, bins = range(0, n_clusters+1))
			hist = hist / np.linalg.norm(hist)
			train_histograms[img] = hist

	elif mode == "val":
		for img, desc in test_features.items():
			preds = kmeans.predict(desc)
			hist, _ = np.histogram(preds, bins = range(0, n_clusters+1))
			hist = hist / np.linalg.norm(hist)
			test_histograms[img] = hist		

		
		

def majority_voting(distances, labels):

    sorted_indices = np.argsort(distances)
    votes = []
    for i in range(0,k):
        index = sorted_indices[i]
        votes.append(labels[index])

    _,pos = np.unique(np.asarray(votes),return_inverse=True)    
    counts = np.bincount(pos) 
    maxpos = counts.argmax() 

    return votes[maxpos]


def knn_test():
	f = open(test_out_file, 'w')
	correct = 0
	total = 0
	for img_name, test_hist in test_histograms.items():
		dists = []
		for train_hist in train_histograms.values():
			dists.append(np.linalg.norm(test_hist-train_hist))

		cluster = majority_voting(dists, train_labels)
		#print("pred for ", img_name, " : ", cluster)
		line = img_name+": "+cluster+"\n"
		f.write(line)




def knn():
	correct = 0
	total = 0
	preds = []
	true_labels = []
	for true_label, test_hist in test_histograms.items():
		
		dists = []
		for train_hist in train_histograms.values():
			dists.append(np.linalg.norm(test_hist-train_hist))

		cluster = majority_voting(dists, train_labels)
		preds.append(cluster)
		true_label = true_label.split('_')

		true_labels.append(true_label[0])
		if true_label[0] == cluster: 
			correct +=1
		total += 1

	#if global varşable show_confusion_matrix set, prints the confusion matrix for the validation set
	if show_confusion_matrix: 
		from sklearn import metrics
		from sklearn import preprocessing

		le = preprocessing.LabelEncoder()
		true_labels1 = le.fit_transform(np.asarray(true_labels))
		le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

		print("mapping: ", le_name_mapping)

		preds1 = le.fit_transform(np.asarray(preds))
		cm = metrics.confusion_matrix(true_labels1, preds1)
		print(cm)

	return correct/total



def SIFT(img):
	# convert to greyscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create(nfeatures= nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)

	#dense SIFT
	if dense: 
		kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
			                                    for x in range(0, gray.shape[1], step_size)]
		dense_feat,descriptors = sift.compute(gray, kp)	
	
	#Normal SIFT		
	else: 
		key_point, descriptors = sift.detectAndCompute(img, None)

    		                                	

	return descriptors





if __name__ == "__main__":

	####################################
	############# TRAIN ################
	####################################

	#load train files and find their descriptors
	train_labels = []	
	for folder in os.listdir(train_path):
		# for each class: folder
		class_name = folder
		class_path = os.path.join(train_path, folder)
		for i, file in enumerate(os.listdir(class_path)):
			train_labels.append(class_name)
			img_path = os.path.join(class_path, file)
			#load image
			img = cv2.imread(img_path)

			# create SIFT feature extractor
			descriptors = SIFT(img)

			#print(img_name)
			img_name = class_name + "_"+ file[:-4]
			if descriptors is not None: 
				train_features[img_name] = descriptors
				for desc in descriptors: all_descriptors.append(desc)


	#print("train descriptors found")



	#apply k means on descriptors
	#print("starting k means")
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.asarray(all_descriptors))
	#print("k means ends")

	#save the res of k_means
	k_means_file_name = "kmeans_"+str(n_clusters)+".pkl"
	with open(k_means_file_name, "wb") as f:
	    pickle.dump(kmeans, f)	

	#build histograms of training imgs
	build_hists("train")
	#print("train hists built")
	#save the built histograms	
	np.save(train_hist_file, list(train_histograms.values()), allow_pickle=True)	    

	
		
	####################################
	############# VALIDATION ###########
	####################################	  


	#load val/test imgs and find local descriptors
	for folder in os.listdir(val_path):
		# for each class: folder
		class_name = folder
		class_path = os.path.join(val_path, folder)
		avg_desc_per_class = 0
		for i, file in enumerate(os.listdir(class_path)): 
			img_path = os.path.join(class_path, file)
			#load image
			img = cv2.imread(img_path)
			# convert to greyscale
			# create SIFT feature extractor
			descriptors = SIFT(img)

			img_name = class_name + "_"+ file[:-4]
			#print("test img: ",img_name)
			if descriptors is not None: 
				#print("none desc for test")
				test_features[img_name] = descriptors

	#build histograms of val imgs
	build_hists("val")	

	#KNN classifier
	acc = knn()
	print()
	print("overall accuracy: ", acc ," with ", n_clusters, " clusters and ", k, " nearest neighbors")
	



	####################################
	############# TEST #################
	####################################
	
	if test == 1:

		for i, file in enumerate(os.listdir(test_path)): 
			img_path = os.path.join(test_path, file)
			#load image
			img = cv2.imread(img_path)
			# convert to greyscale
			# create SIFT feature extractor
			descriptors = SIFT(img)

			img_name = file
			#print("test img: ",img_name)
			if descriptors is not None: 
				#print("none desc for test")
				test_features[img_name] = descriptors

		#build histograms of val imgs
		build_hists("val")	
		#print("test_histograms \n", test_histograms)
		#print()
		knn_test()




	
	"""

	####################################
	#### LOAD TRAINING THEN TEST #######
	####################################

	# load hists
	hists = np.load("train_32means_clusters_sift.npy", allow_pickle=True) # load
	train_labels = []	
	for folder in os.listdir(train_path):
		# for each class: folder
		class_name = folder
		class_path = os.path.join(train_path, folder)
		for i, hist in enumerate(hists):
			#print("hist: \n", hist)
			train_labels.append(class_name)
			img_name = class_name + "_"+ str(i)
			train_histograms[img_name] = hist

	print("train hists loaded")

	#load the kmeans			
	f = open("kmeans_32.pkl", "rb") 
	kmeans =  pickle.load(f)

	print("kmeans loaded")



	#load val/test imgs and find local descriptors
	for folder in os.listdir(val_path):
		# for each class: folder
		class_name = folder
		class_path = os.path.join(val_path, folder)
		avg_desc_per_class = 0
		for i, file in enumerate(os.listdir(class_path)): 
			img_path = os.path.join(class_path, file)
			#load image
			img = cv2.imread(img_path)
			# convert to greyscale
			# create SIFT feature extractor
			descriptors = SIFT(img)

			img_name = class_name + "_"+ file[:-4]
			#print("test img: ",img_name)
			if descriptors is not None: 
				#print("none desc for test")
				test_features[img_name] = descriptors
				

	print("val/test imgs descs found")



	#build histograms of val imgs
	build_hists("val")
	print("val hists built")	
	#print("test_histograms \n", test_histograms)
	#print()


	#KNN classifier
	acc = knn()
	print()
	print("overall acc: ", acc, " with k of kNN ", k)	
	"""
	

	



		
	
	


			
		



